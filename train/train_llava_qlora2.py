#!/usr/bin/env python3
# train_llava_qlora2.py
import os
import math
import random
from pathlib import Path
from typing import Dict, Any
import json
import logging
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,  # alias for newer transformers - will fallback with trust_remote_code below
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

from dataset_llava import LLaVAPairs  # patched dataset file


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_1d(tensors, pad_value: int):
    max_len = max(t.size(0) for t in tensors)
    out = []
    for t in tensors:
        if t.size(0) == max_len:
            out.append(t)
        else:
            pad = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
            out.append(torch.cat([t, pad], dim=0))
    return torch.stack(out, dim=0)


def make_collate(pad_token_id: int):
    def collate(batch: Dict[str, Any]):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                if k == "pixel_values":
                    out[k] = torch.stack(vals, dim=0)
                elif k == "input_ids":
                    out[k] = pad_1d(vals, pad_value=pad_token_id)
                elif k == "attention_mask":
                    out[k] = pad_1d(vals, pad_value=0)
                elif k == "labels":
                    out[k] = pad_1d(vals, pad_value=-100)
                else:
                    out[k] = (
                        torch.stack(vals, dim=0)
                        if all(t.size() == v0.size() for t in vals)
                        else pad_1d(vals, 0)
                    )
            else:
                out[k] = vals
        return out

    return collate


def evaluate(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(**batch)
            total += out.loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def save_checkpoint(accelerator, model, optim, sched, global_step, out_path: Path, logger):
    logger.info(f"Saving checkpoint to {out_path}")
    accelerator.save(
        {
            "model": accelerator.get_state_dict(model),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "global_step": global_step,
        },
        out_path,
    )


def setup_logger(log_file: str = None):
    logger = logging.getLogger("train_llava")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_answer_len", type=int, default=4096)
    ap.add_argument("--sanity_first_batch", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--log_file", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.log_file if args.log_file else None)
    seed_all(42)

    # use string for mixed_precision arg to Accelerator for compatibility
    mixed_precision = "bf16" if args.bf16 else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # newer accelerate exposes chosen mixed precision via accelerator.state.mixed_precision
    mp = getattr(accelerator.state, "mixed_precision", None)
    logger.info(f"[accelerate] device={accelerator.device} mixed_precision={mp}")

    # ----------------------
    # Tokenizer + ensure <image>
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_token = "<image>"
    img_id = tokenizer.convert_tokens_to_ids(image_token)
    logger.info("DEBUG: <image> token id BEFORE fix: %s", img_id)
    if img_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        img_id = tokenizer.convert_tokens_to_ids(image_token)
        logger.info("DEBUG: <image> token ADDED. New ID: %s", img_id)

    # ----------------------
    # Processor (tokenizer + image processor) - use AutoProcessor for correct behavior
    # ----------------------
    logger.info("Loading AutoProcessor (will handle tokenization + image preprocessing)...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # If tokenizer updated above (added special tokens), ensure processor.tokenizer uses same tokenizer object
    # (AutoProcessor stores its own tokenizer; replace it with our tokenizer to be consistent)
    processor.tokenizer = tokenizer

    # ----------------------
    # Model load and resize embeddings
    # ----------------------
    logger.info("Loading base model in 4-bit ...")
    # prefer AutoModelForImageTextToText (newer name). If that import is not available due to older transformers,
    # AutoModelForVision2Seq will be available via trust_remote_code and model_name.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
        )
    except Exception:
        # fallback to generic from_pretrained using trust_remote_code (older code paths)
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
        )

    # resize embeddings after tokenizer changes
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lcfg)

    # ----------------------
    # Dataset
    # ----------------------
    full_ds = LLaVAPairs(
        json_path=args.train_json,
        images_root=args.images_root,
        processor=processor,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # optional subsample
    if args.max_samples and args.max_samples < len(full_ds):
        full_ds, _ = random_split(
            full_ds, [args.max_samples, len(full_ds) - args.max_samples]
        )

    # Validate dataset items (after dataset's internal tokenization/padding)
    logger.info("Validating dataset for <image> token correctness and label lengths...")
    bad = []
    tok = tokenizer
    for i in range(len(full_ds.items)):
        it = full_ds.items[i]
        first = it["conversations"][0]["value"]
        ids = tok(first, add_special_tokens=False)["input_ids"]
        c = ids.count(img_id)
        if c < 1:
            bad.append((i, "image_token_missing", c, first[:200]))

        ans = it["conversations"][1]["value"] if len(it["conversations"]) > 1 else ""
        lab_ids = tok(ans, add_special_tokens=False)["input_ids"]
        if len(lab_ids) > args.max_answer_len:
            bad.append((i, "answer_too_long", len(lab_ids), first[:120]))

    if bad:
        logger.error("Found invalid dataset samples (index, issue, count, preview):")
        for b in bad[:50]:
            logger.error(b)
        raise RuntimeError(
            "Dataset validation failed. Fix samples reported above (ensure at least one <image> token and answer token length <= max_answer_len)."
        )

    logger.info("Dataset OK. Every prompt has at least one <image> token and answer sizes are within limit.")

    # ----------------------
    # DataLoaders
    # ----------------------
    val_size = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.train_bs // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # ----------------------
    # Optim + schedule
    # ----------------------
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.train_bs))
    total_updates = args.epochs * steps_per_epoch // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_updates)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_updates)

    model, optim, train_loader, val_loader, sched = accelerator.prepare(
        model, optim, train_loader, val_loader, sched
    )

    # ----------------------
    # Training loop (batch checks)
    # ----------------------
    logger.info(
        f"Train size: {train_size}  Val size: {val_size}  Steps/epoch: {steps_per_epoch}"
    )
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            # move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(accelerator.device, non_blocking=True)

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # quick sanity checks for image token counts
            img_counts = (input_ids == img_id).sum(dim=1).tolist()
            if any(c == 0 for c in img_counts):
                # log detailed debug info for the bad samples
                logger.error(f"Bad batch: missing <image> token in samples = {img_counts}")
                # if possible decode first sample to inspect
                try:
                    dec = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                except Exception:
                    dec = "<cannot decode>"
                logger.error("Example decoded input_ids (sample 0): %s", dec)
                # raise to stop (you can change to continue if you prefer)
                raise RuntimeError(f"Bad batch: missing <image> token in samples = {img_counts}")

            # label length guard
            max_total_len = args.max_prompt_len + args.max_answer_len
            if labels.shape[1] > max_total_len:
                raise RuntimeError(
                    f"Bad batch: label length {labels.shape[1]} exceeds configured limit {max_total_len}"
                )

            out = model(**batch)
            loss = out.loss / args.grad_accum
            accelerator.backward(loss)

            if (step + 1) % args.grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1

                if accelerator.is_main_process and global_step % args.save_steps == 0:
                    ckpt_path = Path(args.output_dir) / f"step_{global_step}.pt"
                    save_checkpoint(accelerator, model, optim, sched, global_step, ckpt_path, logger)

                    # copy to upload folder
                    os.system(f"cp {ckpt_path} /kaggle/working/checkpoints_for_upload/")

                    # upload new version
                    kaggle_upload("/kaggle/working/checkpoints_for_upload", f"step_{global_step}", logger)

    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / "final.pt"
        accelerator.save({"model": accelerator.get_state_dict(model)}, final_path)
        os.system(f"cp {final_path} /kaggle/working/checkpoints_for_upload/")
        kaggle_upload("/kaggle/working/checkpoints_for_upload", "final adapter", logger)


if __name__ == "__main__":
    main()
