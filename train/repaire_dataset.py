import json
import os
from tqdm import tqdm

INPUT_JSON = r"E:\SCREENSHOT2CODE\dataset_combined\train_data_llava_clean.json"
OUTPUT_JSON = r"E:\SCREENSHOT2CODE\dataset_combined\train_data_llava_final.json"
IMAGES_ROOT = r"E:\SCREENSHOT2CODE\dataset_combined"
BAD_LOG = r"E:\SCREENSHOT2CODE\dataset_combined\bad_entries.log"


def is_invalid_text(x):
    if x is None:
        return True
    if not isinstance(x, str):
        return True
    if len(x.strip()) == 0:
        return True
    if "DEBUG" in x:
        return True
    return False


def fix_entry(entry, index):
    """Strict validator for LLaVA format."""
    # ---- Basic Structure Checks ----
    if not isinstance(entry, dict):
        return None, "Not a dict"

    if "image" not in entry:
        return None, "Missing image field"

    if "conversations" not in entry:
        return None, "Missing conversations"

    # ---- Validate image path ----
    image_path = os.path.join(IMAGES_ROOT, entry["image"])
    if not os.path.exists(image_path):
        return None, f"Missing image file: {entry['image']}"

    conv = entry["conversations"]
    if not isinstance(conv, list) or len(conv) < 2:
        return None, "Invalid conversation length"

    user = conv[0]
    assistant = conv[1]

    # ---- Role checks ----
    if user.get("role") != "user":
        return None, "First message is not user role"
    if assistant.get("role") != "assistant":
        return None, "Second message is not assistant role"

    # ---- Content checks ----
    user_text = user.get("content")
    assistant_text = assistant.get("content")

    if is_invalid_text(user_text):
        return None, "Invalid or empty user prompt"

    if is_invalid_text(assistant_text):
        return None, "Invalid or empty assistant response"

    # ---- Must contain image tag ----
    if "<image>" not in user_text:
        user_text = "<image>\n" + user_text.strip()

    # Rebuild cleaned entry
    cleaned = {
        "image": entry["image"],
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    }

    return cleaned, None


def main():
    print("🔍 Loading dataset...")
    data = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

    cleaned = []
    bad_log = []

    print("🧹 Cleaning + strict validation...")
    for idx, entry in enumerate(tqdm(data)):
        fixed, err = fix_entry(entry, idx)
        if fixed:
            cleaned.append(fixed)
        else:
            bad_log.append(f"#{idx}: {err}")

    print(f"✔ Final cleaned samples: {len(cleaned)}")
    print(f"❌ Removed: {len(bad_log)} broken entries")

    print("💾 Saving final dataset…")
    json.dump(cleaned, open(OUTPUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("🧾 Logging bad entries…")
    with open(BAD_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(bad_log))

    print(f"🎉 DONE — Final dataset saved to:\n{OUTPUT_JSON}")
    print(f"🟥 Bad entries recorded at:\n{BAD_LOG}")


if __name__ == "__main__":
    main()
