#!/usr/bin/env python3
"""
train_with_checkpoint_and_tracking.py

Fast, reliable QLoRA training with:
 - periodic full checkpoints (model+optim+sched+step)
 - LoRA adapter saves
 - status.json live updates for external monitoring
 - graceful Ctrl+C saving and resume
 - configurable intervals and safety

Place next to dataset_llava.py and run as your training entry-point.
"""

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime
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
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from dataset_llava import LLaVAPairs  # user-provided dataset file

# -----------------------
# Utilities
# -----------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, out_path: Path):
    """
    Save atomically: write to tmp then move.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(out_path)


def write_json_atomic(data: Dict[str, Any], out_path: Path):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def make_collate(pad_token_id: int):
    def pad_1d(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            if t.size(0) == max_len:
                out.append(t)
            else:
                pad = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
                out.append(torch.cat([t, pad], dim=0))
        return torch.stack(out, dim=0)

    def collate(batch):
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
                    out[k] = torch.stack(vals, dim=0) if all(t.size() == v0.size() for t in vals) else pad_1d(vals, 0)
            else:
                out[k] = vals
        return out

    return collate


# -----------------------
# Main training code
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Train with robust checkpointing + live tracking")
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
    ap.add_argument("--print_interval", type=int, default=20)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_prompt_len", type=int, default=768)
    ap.add_argument("--max_answer_len", type=int, default=1536)
    ap.add_argument("--resume", type=str, default="", help="Path to full checkpoint .pt to resume (model+optim+sched)")
    ap.add_argument("--status_path", type=str, default="status.json", help="Where to write live status for external monitor")
    ap.add_argument("--hf_offline", action="store_true", help="Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / args.status_path

    seed_all(42)

    # set offline flags if requested
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16")
    device = accelerator.device
    accelerator.print(f"[accelerate] device={device} bf16={args.bf16}")

    # tokenizer + image token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    image_token = "<image>"
    img_id = tokenizer.convert_tokens_to_ids(image_token)
    if img_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        img_id = tokenizer.convert_tokens_to_ids(image_token)
    accelerator.print("DEBUG: <image> token id:", img_id)

    # image processor wrapper
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

    # load model (4-bit)
    accelerator.print("Loading base model in 4-bit ... (may read cache; ensure HF cache set if offline)")
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)

    # dataset
    full_ds = LLaVAPairs(
        json_path=args.train_json,
        images_root=args.images_root,
        processor=processor,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    if args.max_samples and args.max_samples < len(full_ds):
        full_ds, _ = random_split(full_ds, [args.max_samples, len(full_ds) - args.max_samples])

    # quick validation
    accelerator.print("Validating dataset items...")
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
        for b in bad[:20]:
            accelerator.print(b)
        raise RuntimeError("Dataset validation failed. Fix dataset samples reported above.")

    accelerator.print("✔ Dataset OK.")

    # dataloaders
    val_size = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.train_bs // 2), shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # optimizer & scheduler
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.train_bs))
    total_updates = args.epochs * steps_per_epoch // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * max(1, total_updates))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max(1, total_updates))

    # prepare accelerator (wraps model/optim/loaders/sched)
    model, optim, train_loader, val_loader, sched = accelerator.prepare(model, optim, train_loader, val_loader, sched)

    # resume support
    global_step = 0
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            accelerator.print(f"[resume] loading {resume_path}")
            ck = torch.load(resume_path, map_location="cpu")
            try:
                # load model state dict (partial loading safe)
                model.load_state_dict(ck["model"], strict=False)
                optim.load_state_dict(ck["optim"])
                sched.load_state_dict(ck["sched"])
                global_step = int(ck.get("global_step", 0))
                start_epoch = int(ck.get("epoch", 0))
                accelerator.print(f"[resume] loaded checkpoint at step {global_step}, epoch {start_epoch}")
            except Exception as e:
                accelerator.print(f"[resume] failed to fully load checkpoint ({e}). You can still use adapter-only resume.")
        else:
            accelerator.print("[resume] path not found, ignoring resume arg")

    # status writer helper
    def write_status(**kwargs):
        data = {
            "time": datetime.utcnow().isoformat() + "Z",
            "global_step": int(global_step),
            "epoch": int(epoch if 'epoch' in locals() else start_epoch),
            "loss": float(kwargs.get("loss", math.nan)),
            "steps_done": int(kwargs.get("steps_done", 0)),
            "total_steps": int(total_updates),
            "eta_hours": float(kwargs.get("eta_hours", math.nan)),
        }
        try:
            write_json_atomic(data, status_path)
        except Exception:
            pass

    # graceful shutdown handler: save current state + LoRA adapter
    stop_requested = False

    def save_everything(tag="manual"):
        nonlocal global_step
        if not accelerator.is_main_process:
            return
        try:
            accelerator.print(f"[save] saving full checkpoint (step={global_step}) ...")
            ck_path = out_dir / f"checkpoint_{tag}_step{global_step}.pt"
            payload = {
                "model": accelerator.get_state_dict(model),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "global_step": global_step,
                "epoch": epoch if 'epoch' in locals() else start_epoch,
            }
            atomic_save(payload, ck_path)
            accelerator.print(f"[save] full checkpoint -> {ck_path}")

            # save LoRA adapter only (smaller)
            lora_path = out_dir / f"lora_adapter_{tag}_step{global_step}.pt"
            adapter_state = {"lora_state_dict": accelerator.get_state_dict(model)}
            atomic_save(adapter_state, lora_path)
            accelerator.print(f"[save] LoRA adapter -> {lora_path}")

            # update status json
            write_status(loss=last_print_loss if 'last_print_loss' in locals() else math.nan, steps_done=global_step, eta_hours=0.0)
        except Exception as e:
            accelerator.print(f"[save] exception while saving: {e}")

    def _signal_handler(sig, frame):
        nonlocal stop_requested
        accelerator.print(f"[signal] caught signal {sig}. Will attempt graceful save and exit.")
        stop_requested = True
        save_everything(tag="signal")
        # allow script to exit after handler; set flag checked in loop

    # register handler (works for KeyboardInterrupt and SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass  # Windows may behave differently

    # training loop
    accelerator.print(f"Train size: {train_size}  Val size: {val_size}  Steps/epoch: {steps_per_epoch}  Total updates: {total_updates}")
    total_steps = max(1, total_updates)
    print_interval = max(1, args.print_interval)

    start_time = time.time()
    last_print_time = start_time
    last_print_loss = math.nan

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            local_steps = 0

            for step, batch in enumerate(train_loader):
                if stop_requested:
                    break

                # move to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device, non_blocking=True)

                # forward/back
                out = model(**batch)
                loss = out.loss / args.grad_accum
                accelerator.backward(loss)

                running_loss += float(loss.detach().cpu().item())
                local_steps += 1

                if (step + 1) % args.grad_accum == 0:
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    # periodic print + status update
                    if global_step % print_interval == 0 or global_step == 1:
                        now = time.time()
                        elapsed = now - start_time
                        steps_done = max(1, global_step)
                        steps_left = max(0, total_steps - global_step)
                        avg_loss = running_loss / max(1, local_steps)
                        # estimate hours remaining by simple linear extrapolation
                        eta_hours = (elapsed / steps_done) * steps_left / 3600.0 if steps_done > 0 else float("nan")
                        accelerator.print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {global_step}/{total_steps} | loss: {avg_loss:.4f} | ETA: {eta_hours:.2f} hrs")
                        last_print_time = now
                        last_print_loss = avg_loss
                        # write status
                        write_status(loss=avg_loss, steps_done=global_step, eta_hours=eta_hours)
                        running_loss = 0.0
                        local_steps = 0

                    # eval block
                    if accelerator.is_main_process and args.eval_steps and global_step % args.eval_steps == 0:
                        val_loss = 0.0
                        n_val = 0
                        model.eval()
                        with torch.no_grad():
                            for vb in val_loader:
                                for k in vb:
                                    if isinstance(vb[k], torch.Tensor):
                                        vb[k] = vb[k].to(device, non_blocking=True)
                                outv = model(**vb)
                                val_loss += float(outv.loss.detach().cpu().item())
                                n_val += 1
                        model.train()
                        val_loss = val_loss / max(1, n_val)
                        accelerator.print(f"[eval] step {global_step} val_loss={val_loss:.4f}")
                        write_status(loss=val_loss, steps_done=global_step, eta_hours=0.0)

                    # periodic checkpoint save
                    if accelerator.is_main_process and args.save_steps and global_step % args.save_steps == 0:
                        save_everything(tag=f"step{global_step}")

                # end grad-accum block

            epoch_minutes = (time.time() - epoch_start) / 60.0
            accelerator.print(f"⏳ Epoch {epoch+1} finished in {epoch_minutes:.2f} minutes")
            # write epoch-level status
            write_status(loss=last_print_loss if 'last_print_loss' in locals() else math.nan, steps_done=global_step, eta_hours=0.0)

            if stop_requested:
                accelerator.print("[main] Stop requested; leaving training loop.")
                break

        # final save on normal completion
        if accelerator.is_main_process:
            save_everything(tag="final")
    except Exception as e:
        accelerator.print(f"[ERROR] training loop exception: {e}")
        # try final save on unhandled exception
        try:
            save_everything(tag="error")
        except Exception:
            pass
        raise
    finally:
        accelerator.print("Training finished (exiting).")
        # ensure status file indicates finished
        write_status(loss=last_print_loss if 'last_print_loss' in locals() else math.nan, steps_done=global_step, eta_hours=0.0)

if __name__ == "__main__":
    main()
