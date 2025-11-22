#!/usr/bin/env python3
"""
train_timed_verbose_fixed.py

- Loads dataset in your provided JSON format (id, image, conversations).
- Prints every batch and every optimizer step.
- Saves LoRA adapter every --save_every_mins minutes (and also periodic full checkpoints optionally).
- Graceful Ctrl+C saves adapter before exiting.
"""

import argparse
import json
import os
import time
from pathlib import Path
import signal
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForVision2Seq, get_cosine_schedule_with_warmup
from accelerate import Accelerator

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict

# -------------------------
# Dataset for your JSON
# -------------------------
class LLaVADataset(Dataset):
    def __init__(self, json_path, images_root, processor, max_prompt_len=768, max_answer_len=1536):
        self.images_root = Path(images_root)
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.items = items
        self.processor = processor
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        # image path
        img_path = self.images_root / entry["image"]
        # conversations: human first, gpt second (as per sample)
        convs = entry.get("conversations", [])
        if len(convs) < 1:
            raise RuntimeError(f"Entry {idx} missing conversations")
        # Find human prompt and gpt answer robustly:
        human_val = None
        gpt_val = ""
        # prefer first human and first subsequent gpt
        for c in convs:
            if c.get("from") == "human" and human_val is None:
                human_val = c.get("value", "")
            elif c.get("from") == "gpt" and gpt_val == "":
                gpt_val = c.get("value", "")

        if human_val is None:
            raise RuntimeError(f"Entry {idx} missing human prompt")

        # expected prompt contains the <image> token (your data shows "<image>\n...")
        prompt_text = human_val
        answer_text = gpt_val

        # Tokenize (processor should produce pixel_values and we call tokenizer directly)
        # processor is SimpleProcessor-like: processor(images, text, max_length=..., return_tensors="pt")
        processed = self.processor([str(img_path)], prompt_text, padding="longest", truncation=True, max_length=self.max_prompt_len, return_tensors="pt")
        # For labels we tokenize the answer (no special <image> there)
        # Use tokenizer attribute on processor (SimpleProcessor below exposes tokenizer)
        tok = self.processor.tokenizer
        lab = tok(answer_text, padding="longest", truncation=True, max_length=self.max_answer_len, return_tensors="pt")
        labels = lab["input_ids"].squeeze(0)
        # convert tensors to expected shapes (no batch dim here)
        out = {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": labels,
        }
        return out

# -------------------------
# collate fn
# -------------------------
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
    def collate(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                if k == "pixel_values":
                    out[k] = torch.stack(vals, dim=0)
                elif k in ("input_ids", "attention_mask", "labels"):
                    pad_id = pad_token_id if k == "input_ids" else (0 if k=="attention_mask" else -100)
                    out[k] = pad_1d(vals, pad_value=pad_id)
                else:
                    out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out
    return collate

# -------------------------
# SimpleProcessor (wrap tokenizer + image_processor)
# -------------------------
class SimpleProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, image_paths, text, padding="max_length", truncation=True, max_length=None, return_tensors="pt"):
        # image_paths: list of file paths
        imgs = [torch.from_numpy(self.image_processor(Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"].numpy()).squeeze(0) if False else None]
        # But above is awkward in offline env; instead let image_processor handle list directly:
        # (transformers AutoImageProcessor accepts PIL images or file paths depending; safe approach: open PIL)
        from PIL import Image
        pil_imgs = []
        for p in image_paths:
            p2 = Path(p)
            if not p2.exists():
                # try absolute path as fallback
                p2 = Path(p)
            img = Image.open(p2).convert("RGB")
            pil_imgs.append(img)
        pix = self.image_processor(pil_imgs, return_tensors="pt")["pixel_values"]
        toks = self.tokenizer(text, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
        return {"pixel_values": pix, "input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

# -------------------------
# Save helper (LoRA adapter only + full)
# -------------------------
def save_adapter_only(model, out_path: Path, accelerator: Accelerator):
    # get peft state dict (adapter weights)
    try:
        state = get_peft_model_state_dict(model)
    except Exception:
        # fallback: save full model state
        state = accelerator.get_state_dict(model)
    accelerator.save(state, str(out_path))

def save_full_checkpoint(accelerator, model, optim, sched, global_step, out_path: Path):
    accelerator.save({
        "model": accelerator.get_state_dict(model),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "global_step": global_step
    }, str(out_path))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--print_interval", type=int, default=1, help="print every N optimizer steps")
    ap.add_argument("--save_every_mins", type=float, default=30.0, help="timed adapter save in minutes")
    ap.add_argument("--save_full_every_mins", type=float, default=0.0, help="timed full checkpoint save in minutes (0 disables)")
    ap.add_argument("--max_prompt_len", type=int, default=768)
    ap.add_argument("--max_answer_len", type=int, default=1536)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16")
    device = accelerator.device
    accelerator.print(f"[accelerate] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # ensure <image> token exists (common pattern)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    image_processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor = SimpleProcessor(tokenizer, image_processor)

    accelerator.print("Loading base model (4-bit friendly)...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lcfg)

    # dataset
    full_ds = LLaVADataset(args.train_json, args.images_root, processor, args.max_prompt_len, args.max_answer_len)

    # train/val split (small val)
    val_size = max(1, int(0.02 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.train_bs//2), shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(train_size / max(1, args.train_bs))
    total_updates = args.epochs * steps_per_epoch // max(1, args.grad_accum)
    warmup_steps = int(0.03 * max(1, total_updates))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max(1, total_updates))

    model, optim, train_loader, val_loader, sched = accelerator.prepare(model, optim, train_loader, val_loader, sched)

    # resume support: if resume file given, try load adapter weights
    start_global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        try:
            model.load_state_dict(ck["model"], strict=False)
            optim.load_state_dict(ck["optim"])
            sched.load_state_dict(ck["sched"])
            start_global_step = int(ck.get("global_step", 0))
            accelerator.print(f"[resume] loaded resume checkpoint {args.resume} at step {start_global_step}")
        except Exception as e:
            accelerator.print(f"[resume] failed to load full checkpoint ({e}). Continuing from scratch.")

    # Ctrl+C handler: save adapter immediately
    stop_requested = {"flag": False}
    def _signal_handler(sig, frame):
        stop_requested["flag"] = True
        accelerator.print("SIGINT received — will stop after current optimizer step and save adapter.")
    signal.signal(signal.SIGINT, _signal_handler)

    # training loop
    global_step = start_global_step
    total_steps = max(1, total_updates)

    last_timed_save = time.time()
    last_full_save = time.time()
    save_every_secs = args.save_every_mins * 60.0
    save_full_secs = args.save_full_every_mins * 60.0 if args.save_full_every_mins>0 else None

    accelerator.print(f"Train size={train_size}, Val size={val_size}, Total updates={total_updates}")
    running_loss = 0.0
    local_step_count = 0
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_start = time.time()
            for step, batch in enumerate(train_loader):
                # move tensors done by accelerator
                # Print batch shapes (very verbose)
                if accelerator.is_local_main_process:
                    accelerator.print(f"\n===== EPOCH {epoch+1} BATCH {step} =====")
                    for k,v in batch.items():
                        if isinstance(v, torch.Tensor):
                            accelerator.print(f"{k}: {tuple(v.shape)} {v.dtype}")

                out = model(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()})
                loss = out.loss / args.grad_accum
                accelerator.backward(loss)
                running_loss += loss.item()
                local_step_count += 1

                if (step + 1) % args.grad_accum == 0:
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    # print interval (per optimizer step)
                    if global_step % max(1,args.print_interval) == 0:
                        elapsed = time.time() - start_time
                        avg_loss = running_loss / max(1, local_step_count)
                        steps_done = global_step
                        steps_left = max(0, total_steps - steps_done)
                        eta_hours = (elapsed / max(1,steps_done)) * steps_left / 3600.0
                        accelerator.print(f"[{time.strftime('%H:%M:%S')}] Step {global_step}/{total_steps} | loss: {avg_loss:.6f} | ETA: {eta_hours:.3f} hrs")
                        running_loss = 0.0
                        local_step_count = 0

                    # timed adapter save
                    now = time.time()
                    if save_every_secs and (now - last_timed_save) >= save_every_secs and accelerator.is_main_process:
                        last_timed_save = now
                        adapter_path = out_dir / f"adapter_step_{global_step}.pt"
                        save_adapter_only(model, adapter_path, accelerator)
                        accelerator.print(f"💾 LoRA adapter saved to {adapter_path}")

                    # timed full checkpoint save
                    if save_full_secs and (now - last_full_save) >= save_full_secs and accelerator.is_main_process:
                        last_full_save = now
                        full_path = out_dir / f"full_ck_step_{global_step}.pt"
                        save_full_checkpoint(accelerator, model, optim, sched, global_step, full_path)
                        accelerator.print(f"💾 Full checkpoint saved to {full_path}")

                    # early stop if user requested
                    if stop_requested["flag"]:
                        accelerator.print("User requested stop. Saving adapter and exiting training loop.")
                        if accelerator.is_main_process:
                            adapter_path = out_dir / f"adapter_step_{global_step}_on_cancel.pt"
                            save_adapter_only(model, adapter_path, accelerator)
                            accelerator.print(f"💾 Adapter saved: {adapter_path}")
                        raise KeyboardInterrupt

            epoch_time = (time.time() - epoch_start) / 60.0
            accelerator.print(f"⏳ Epoch {epoch+1} done in {epoch_time:.2f} min")

    except KeyboardInterrupt:
        accelerator.print("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        accelerator.print(f"Training exception: {e}")
        raise
    finally:
        # final save adapter + optionally full
        if accelerator.is_main_process:
            final_adapter = out_dir / f"final_adapter_step_{global_step}.pt"
            save_adapter_only(model, final_adapter, accelerator)
            accelerator.print(f"✔ Final LoRA adapter saved: {final_adapter}")
            final_full = out_dir / f"final_full_step_{global_step}.pt"
            save_full_checkpoint(accelerator, model, optim, sched, global_step, final_full)
            accelerator.print(f"✔ Final full checkpoint saved: {final_full}")

    accelerator.print("Training finished.")

if __name__ == "__main__":
    main()
