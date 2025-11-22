#!/usr/bin/env python3
"""
validate_dataset.py

Usage example:
python validate_dataset.py \
  --json "E:\SCREENSHOT2CODE\dataset_combined\train_data_fixed.json" \
  --images_root "E:\SCREENSHOT2CODE\dataset_combined" \
  --max_answer_len 4096 \
  --report "E:\SCREENSHOT2CODE\checkpoints\fast_verbose_logs\dataset_report.json" \
  --csv "E:\SCREENSHOT2CODE\checkpoints\fast_verbose_logs\dataset_report.csv"

Optional: pass --tokenizer_model to validate answer token lengths using a HF tokenizer.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import csv

def normalize_prompt(p: str) -> str:
    if p is None:
        return ""
    s = p.strip()
    s = s.replace("< image >", "<image>")
    s = s.replace("<Image>", "<image>")
    s = s.replace("<IMG>", "<image>")
    s = s.replace("<Image >", "<image>")
    return s

def resolve_image_path(images_root: Path, rel: str) -> Path:
    # reproduce the same resolution rules as dataset_llava
    rel_lower = rel.lower()
    root = images_root
    if root.name.lower() == "images" and rel_lower.startswith("images/"):
        rel2 = rel[len("images/"):]
        return root / rel2
    if rel_lower.startswith("images/"):
        return root / rel
    return root / "images" / rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to dataset json")
    ap.add_argument("--images_root", required=True, help="Images root dir")
    ap.add_argument("--max_answer_len", type=int, default=4096, help="Max answer token length (or chars if no tokenizer)")
    ap.add_argument("--report", default="dataset_report.json", help="JSON report output path")
    ap.add_argument("--csv", default="", help="CSV report output path (optional)")
    ap.add_argument("--tokenizer_model", default="", help="HuggingFace tokenizer model to check token lengths (optional)")
    ap.add_argument("--max_preview", type=int, default=120, help="Preview length for console output")
    args = ap.parse_args()

    js_path = Path(args.json)
    if not js_path.exists():
        print(f"ERROR: dataset json not found: {js_path}", file=sys.stderr)
        sys.exit(2)

    images_root = Path(args.images_root)
    if not images_root.exists():
        print(f"ERROR: images_root not found: {images_root}", file=sys.stderr)
        sys.exit(2)

    data = json.loads(js_path.read_text(encoding="utf-8"))
    total = len(data)
    print(f"Loaded {total} items from {js_path}")

    # optional tokenizer
    tokenizer = None
    if args.tokenizer_model:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False)
            print(f"Loaded tokenizer: {args.tokenizer_model}")
        except Exception as e:
            print(f"Failed to load tokenizer '{args.tokenizer_model}': {e}. Continuing without tokenizer.", file=sys.stderr)
            tokenizer = None

    problems: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    first_bad_previewed = 0

    for idx, it in enumerate(data):
        item_report: Dict[str, Any] = {"index": idx}
        img_rel = it.get("image", "")
        try:
            img_path = resolve_image_path(images_root, img_rel)
        except Exception as e:
            img_path = images_root / img_rel
        item_report["image_path"] = str(img_path)
        if not Path(img_path).exists():
            item_report["image_exists"] = False
            item_report["issue"] = item_report.get("issue", []) + ["image_not_found"]
        else:
            item_report["image_exists"] = True

        conv = it.get("conversations", [])
        if not isinstance(conv, list) or len(conv) == 0:
            item_report["issue"] = item_report.get("issue", []) + ["no_conversations"]
            conv = [{"from":"human","value":""}]

        raw_prompt = conv[0].get("value", "") if len(conv) > 0 else ""
        raw_prompt = normalize_prompt(raw_prompt)
        item_report["raw_prompt_preview"] = raw_prompt[: args.max_preview]
        # ensure there's at least one <image> at start — follow your dataset logic
        if "<image>" not in raw_prompt:
            # dataset prep would have injected one
            injected = "<image>\n" + raw_prompt
            raw_prompt = injected
            item_report["injected_image"] = True
        else:
            # keep last section after last <image> like dataset
            parts = raw_prompt.split("<image>")
            raw_prompt = "<image>" + parts[-1].strip()

        item_report["normalized_prompt_preview"] = raw_prompt[: args.max_preview]

        # count image tokens (literal)
        token_count = raw_prompt.count("<image>")
        item_report["image_token_count"] = token_count
        if token_count != 1:
            item_report["issue"] = item_report.get("issue", []) + ["image_token_count!=1"]

        # answer
        answer = ""
        if len(conv) > 1:
            answer = conv[1].get("value", "")
        item_report["answer_preview"] = (answer or "")[: args.max_preview]

        # check answer length: tokenized if tokenizer provided, else measured in characters
        if tokenizer:
            try:
                toks = tokenizer(answer, add_special_tokens=False)
                alen = len(toks["input_ids"])
                item_report["answer_token_len"] = alen
                if alen > args.max_answer_len:
                    item_report["issue"] = item_report.get("issue", []) + ["answer_token_len_gt_max"]
            except Exception as e:
                item_report["issue"] = item_report.get("issue", []) + [f"tokenizer_error:{str(e)[:80]}"]
        else:
            alen = len(answer)
            item_report["answer_char_len"] = alen
            if alen > args.max_answer_len:
                item_report["issue"] = item_report.get("issue", []) + ["answer_char_len_gt_max"]

        if "issue" in item_report:
            problems.append(item_report)
            if first_bad_previewed < 20:
                print("---- BAD ITEM ----")
                print(f"index={idx} image={img_rel} image_exists={item_report.get('image_exists')} image_token_count={token_count}")
                print("prompt preview:", item_report["normalized_prompt_preview"])
                print("answer preview:", item_report["answer_preview"])
                print("issues:", item_report["issue"])
                first_bad_previewed += 1

        report_rows.append(item_report)

    summary = {
        "total": total,
        "problems": len(problems),
        "checked": total,
    }

    # write JSON report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "items": report_rows, "bad_items": problems}
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON report to {report_path}")

    # optional CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["index", "image_path", "image_exists", "image_token_count", "answer_char_len", "answer_token_len", "issue"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in report_rows:
                writer.writerow({
                    "index": r.get("index"),
                    "image_path": r.get("image_path"),
                    "image_exists": r.get("image_exists"),
                    "image_token_count": r.get("image_token_count"),
                    "answer_char_len": r.get("answer_char_len"),
                    "answer_token_len": r.get("answer_token_len"),
                    "issue": ";".join(r.get("issue", [])) if r.get("issue") else ""
                })
        print(f"Wrote CSV report to {csv_path}")

    # final summary print
    print("SUMMARY:")
    print(f"  Total items checked: {total}")
    print(f"  Problematic items:   {len(problems)}")
    if len(problems) > 0:
        print("Problems types (sample):")
        # collect counts
        freq: Dict[str,int] = {}
        for p in problems:
            for iss in p.get("issue", []):
                freq[iss] = freq.get(iss, 0) + 1
        for k,v in sorted(freq.items(), key=lambda x:-x[1]):
            print(f"   - {k}: {v}")

    if len(problems) > 0:
        print("\nExit code 1 returned (problems found). Fix items, re-run.")
        sys.exit(1)
    else:
        print("\nNo problems found. Dataset looks clean.")
        sys.exit(0)

if __name__ == "__main__":
    main()
