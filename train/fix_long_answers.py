import json
from pathlib import Path
from transformers import AutoTokenizer

INPUT = r"E:\SCREENSHOT2CODE\dataset_combined\train_data.json"
OUTPUT = r"E:\SCREENSHOT2CODE\dataset_combined\train_data_fixed.json"
MODEL = "llava-hf/llava-1.5-7b-hf"
MAX_TOK = 1536  # match your training max_answer_len

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False, trust_remote_code=True)

data = json.loads(Path(INPUT).read_text(encoding="utf-8"))
fixed = []
trimmed = 0

for ex in data:
    conv = ex.get("conversations", [])
    if len(conv) < 2:
        fixed.append(ex)
        continue

    answer = conv[1]["value"]
    ids = tok(answer, add_special_tokens=False)["input_ids"]

    if len(ids) > MAX_TOK:
        trimmed += 1
        new_ids = ids[:MAX_TOK]
        new_answer = tok.decode(new_ids, skip_special_tokens=True)
        conv[1]["value"] = new_answer

    fixed.append(ex)

Path(OUTPUT).write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Done. Trimmed {trimmed} long answers.")
print("Saved:", OUTPUT)
