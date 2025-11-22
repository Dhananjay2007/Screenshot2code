#!/usr/bin/env python3
"""
Optimized LLaVA QLoRA training script, integrated and tuned for low-VRAM GPUs (RTX 4050).
Replace your current train_llava_qlora.py with this file.
Keep dataset_llava.py in the same folder.
"""

import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

from dataset_llava import LLaVAPairs


# -----------------------
# Utilities
# -----------------------
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


def evaluate(model, val_loader, device, accelerator):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            out = model(**batch)
            total += out.loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def save_checkpoint_state(accelerator, model, optim, sched, global_step, out_path: Path):
    accelerator.save(
        {
            "model": accelerator.get_state_dict(model),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "global_step": global_step,
        },
        out_path,
    )


# -----------------------
# Main
# -----------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
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
    ap.add_argument("--max_prompt_len", type=int, default=768)
    ap.add_argument("--max_answer_len", type=int, default=1536)
    ap.add_argument("--sanity_first_batch", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--hf_offline", action="store_true", help="Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE")
    ap.add_argument("--print_interval", type=int, default=20)
    ap.add_argument("--early_stop_patience_steps", type=int, default=3000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(42)

    # HF offline flags (optional)
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Accelerator
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16")
    device = accelerator.device
    accelerator.print(f"[accelerate] device={device} bf16={args.bf16}")

    # ----------------------
    # Tokenizer & image token
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_token = "<image>"
    try:
        img_id = tokenizer.convert_tokens_to_ids(image_token)
    except Exception:
        img_id = None
    accelerator.print("DEBUG: <image> token id BEFORE fix:", img_id)

    if img_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        img_id = tokenizer.convert_tokens_to_ids(image_token)
        accelerator.print("DEBUG: <image> token ADDED. New ID:", img_id)

    # ----------------------
    # Image processor and simple processor
    # ----------------------
    image_processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    class SimpleProcessor:
        def __init__(self, tok, imgp):
            self.tokenizer = tok
            self.image_processor = imgp

        def __call__(self, images, text, padding="max_length", truncation=True, max_length=None, return_tensors="pt"):
            pix = self.image_processor(images, return_tensors="pt")["pixel_values"]
            toks = self.tokenizer(text, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
            return {"pixel_values": pix, "input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

    processor = SimpleProcessor(tokenizer, image_processor)

    # ----------------------
    # Load model in 4-bit
    # ----------------------
    accelerator.print("Loading base model in 4-bit ...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )

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

    if args.max_samples and args.max_samples < len(full_ds):
        full_ds, _ = random_split(full_ds, [args.max_samples, len(full_ds) - args.max_samples])

    # Dataset validation (conservative)
    accelerator.print("Validating dataset for <image> token correctness and label lengths...")
    bad = []
    for i in range(len(full_ds.items)):
        it = full_ds.items[i]
        first = it["conversations"][0]["value"]
        ids = tokenizer(first, add_special_tokens=False)["input_ids"]
        c = ids.count(img_id)
        if c != 1:
            bad.append((i, "image_token_count", c, first[:200]))

        ans = it["conversations"][1]["value"] if len(it["conversations"]) > 1 else ""
        lab_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
        if len(lab_ids) > args.max_answer_len:
            bad.append((i, "answer_too_long", len(lab_ids), ans[:120]))

    if bad:
        accelerator.print("Found invalid dataset samples (index, issue, count, preview):")
        for b in bad[:50]:
            accelerator.print(b)
        raise RuntimeError("Dataset validation failed. Fix samples reported above (ensure 1 <image> token and answer token length <= max_answer_len).")

    accelerator.print("✔ Dataset OK. Every prompt has exactly 1 <image> token and answer sizes are within limit.")

    # ----------------------
    # Dataloaders
    # ----------------------
    val_size = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.train_bs // 2), shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # ----------------------
    # Optimizer + scheduler
    # ----------------------
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.train_bs))
    total_updates = args.epochs * steps_per_epoch // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * max(1, total_updates))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_updates)

    # Prepare Accelerator (wraps model, optim, loaders, scheduler)
    model, optim, train_loader, val_loader, sched = accelerator.prepare(model, optim, train_loader, val_loader, sched)

    # Resume support
    start_global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        try:
            model.load_state_dict(ck["model"], strict=False)
            optim.load_state_dict(ck["optim"])
            sched.load_state_dict(ck["sched"])
            start_global_step = int(ck.get("global_step", 0))
            accelerator.print(f"[resume] loaded {args.resume} at global_step={start_global_step}")
        except Exception as e:
            accelerator.print(f"[resume] failed to load full checkpoint ({e}). Continue from scratch.")

    # Optional quick sanity batch
    if args.sanity_first_batch:
        model.eval()
        first = next(iter(train_loader))
        accelerator.print("SANITY batch:", {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)) for k,v in first.items()})
        try:
            image_id = tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            image_id = None
        if image_id is not None and isinstance(first["input_ids"], torch.Tensor):
            counts = (first["input_ids"] == image_id).sum(dim=1).tolist()
            for i, c in enumerate(counts):
                if c != 1:
                    raise RuntimeError(f"Sanity failed: sample {i} has {c} image tokens; expected 1")
        with torch.no_grad():
            for k in first:
                if isinstance(first[k], torch.Tensor):
                    first[k] = first[k].to(device, non_blocking=True)
            out = model(**first)
            accelerator.print(f"SANITY forward ok; loss={float(out.loss):.4f}")
        model.train()

    # ----------------------
    # Optimized training loop
    # ----------------------
    accelerator.print(f"Train size: {train_size}  Val size: {val_size}  Steps/epoch: {steps_per_epoch}")
    global_step = start_global_step
    total_steps = max(1, total_updates)

    best_val_loss = float("inf")
    no_improve_steps = 0
    early_stop_patience = args.early_stop_patience_steps

    print_interval = max(1, args.print_interval)

    start_time = time.time()
    last_print = start_time

    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            local_step_count = 0

            for step, batch in enumerate(train_loader):
                # move tensors
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device, non_blocking=True)

                # quick batch sanity: ensure exactly 1 image token per sample
                try:
                    input_ids = batch["input_ids"]
                    img_counts = (input_ids == img_id).sum(dim=1).tolist()
                    if any(c != 1 for c in img_counts):
                        raise RuntimeError(f"Bad batch: image token counts per sample = {img_counts}")
                except Exception:
                    # if tokenizer mapping changed, skip check (defensive)
                    pass

                # forward/backward
                out = model(**batch)
                loss = out.loss / args.grad_accum
                accelerator.backward(loss)

                running_loss += loss.item()
                local_step_count += 1

                # optimizer step
                if (step + 1) % args.grad_accum == 0:
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    # printing
                    if global_step % print_interval == 0:
                        now = time.time()
                        avg_loss = running_loss / max(1, local_step_count)
                        elapsed = now - start_time
                        steps_done = global_step if global_step > 0 else 1
                        steps_left = max(0, total_steps - global_step)
                        eta_hours = (elapsed / steps_done) * steps_left / 3600.0
                        accelerator.print(f"[{time.strftime('%H:%M:%S')}] Step {global_step}/{total_steps} | loss: {avg_loss:.4f} | ETA: {eta_hours:.2f} hrs")
                        running_loss = 0.0
                        local_step_count = 0

                    # evaluation
                    if accelerator.is_main_process and args.eval_steps and global_step % args.eval_steps == 0:
                        val_loss = evaluate(model, val_loader, device, accelerator)
                        accelerator.print(f"[eval] step {global_step} val_loss={val_loss:.4f}")

                        # save best
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            no_improve_steps = 0
                            best_path = Path(args.output_dir) / "best_checkpoint.pt"
                            save_checkpoint_state(accelerator, model, optim, sched, global_step, best_path)
                            accelerator.print(f"🏆 Saved new best checkpoint to {best_path}")
                        else:
                            no_improve_steps += args.eval_steps
                            if no_improve_steps >= early_stop_patience:
                                accelerator.print("🛑 Early stopping triggered (no improvement).")
                                raise StopIteration

                    # checkpoint saving
                    if accelerator.is_main_process and args.save_steps and global_step % args.save_steps == 0:
                        ck = Path(args.output_dir) / f"step_{global_step}.pt"
                        save_checkpoint_state(accelerator, model, optim, sched, global_step, ck)
                        accelerator.print(f"💾 Saved periodic checkpoint at {ck}")

                # end of batch

            epoch_time = (time.time() - epoch_start) / 60.0
            accelerator.print(f"⏳ Epoch {epoch+1} done in {epoch_time:.2f} min")

        # End training normally
    except StopIteration:
        accelerator.print("Training stopped early by early-stopping.")
    except Exception as e:
        accelerator.print(f"Training loop exception: {e}")
        raise
    finally:
        # final save (main process only)
        if accelerator.is_main_process:
            final_path = Path(args.output_dir) / "final_adapter.pt"
            save_checkpoint_state(accelerator, model, optim, sched, global_step, final_path)
            accelerator.print(f"Done. Final checkpoint saved: {final_path}")

    accelerator.print("Training finished.")


if __name__ == "__main__":
    main()
