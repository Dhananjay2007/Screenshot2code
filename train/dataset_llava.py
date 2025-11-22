# dataset_llava.py
import json
from pathlib import Path
from typing import Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_TOKEN = "<image>"


def _find_subsequence(seq: List[int], subseq: List[int]) -> int:
    """Return start index of subseq in seq or -1 if not found."""
    if not subseq or not seq:
        return -1
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


class LLaVAPairs(Dataset):
    def __init__(
        self,
        json_path: str,
        images_root: str,
        processor,
        max_prompt_len: int = 1024,
        max_answer_len: int = 4096,
    ):
        """
        processor must be an AutoProcessor (tokenizer + image_processor)
        """
        self.items: List[Dict] = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.images_root = Path(images_root)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len

        # Normalize: ensure conversations exist and are two entries
        cleaned = []
        for it in self.items:
            conv = it.get("conversations", [])
            if len(conv) < 2:
                conv = conv + [{"from": "gpt", "value": ""}] * (2 - len(conv))
            it["conversations"] = conv[:2]
            cleaned.append(it)
        self.items = cleaned

        # ensure tokenizer has IMAGE_TOKEN as special token
        img_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if img_id is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            # Note: caller must call model.resize_token_embeddings(len(tokenizer)) after tokenizer update

    def __len__(self):
        return len(self.items)

    def _resolve_image_path(self, rel: str) -> Path:
        root = self.images_root
        # handle common layouts
        if root.name.lower() == "images" and rel.lower().startswith("images/"):
            rel = rel[len("images/") :]
            return root / rel
        if rel.lower().startswith("images/"):
            return root / rel
        return root / "images" / rel

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = self._resolve_image_path(it["image"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (item index {idx})")

        image = Image.open(img_path).convert("RGB")
        conv = it["conversations"]
        raw_prompt = conv[0]["value"].strip()
        answer = conv[1]["value"]

        # Normalise variants of image token
        raw_prompt = raw_prompt.replace("< image >", "<image>")
        raw_prompt = raw_prompt.replace("<Image>", "<image>")
        raw_prompt = raw_prompt.replace("<IMG>", "<image>")

        # Ensure exactly one <image> token present in the textual prompt (keep only last content after token)
        if "<image>" not in raw_prompt:
            raw_prompt = "<image>\n" + raw_prompt
        else:
            parts = raw_prompt.split("<image>")
            raw_prompt = "<image>" + parts[-1].strip()

        # Use the processor to get pixel_values + tokenizer outputs consistently
        # Some processors combine tokenizer+image_processor. Call processor(...) directly.
        enc = self.processor(
            images=image,
            text=raw_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)  # [L]
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = enc["pixel_values"].squeeze(0)

        # Defensive fix: ensure tokenizer produced exactly one image token id
        tok = self.tokenizer
        img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
        if img_id is None:
            # Shouldn't happen; ensure we have it
            tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)

        count_img = int((input_ids == img_id).sum().item())
        if count_img == 0:
            # Attempt automatic correction:
            # 1) If tokenizer splits '<image>' into token sequence, find that subsequence and replace with single img_id.
            seq = tok(IMAGE_TOKEN, add_special_tokens=False)["input_ids"]
            seq_len = len(seq)
            ids_list = input_ids.tolist()
            if seq_len > 1:
                pos = _find_subsequence(ids_list, seq)
                if pos != -1:
                    # replace subsequence with single img_id and pad/truncate to original length
                    new_ids = ids_list[:pos] + [img_id] + ids_list[pos + seq_len :]
                    # trim or extend to match original length
                    L = input_ids.size(0)
                    if len(new_ids) < L:
                        new_ids = new_ids + [tok.pad_token_id or 0] * (L - len(new_ids))
                    else:
                        new_ids = new_ids[:L]
                    input_ids = torch.tensor(new_ids, dtype=torch.long)
                    count_img = int((input_ids == img_id).sum().item())

            # 2) If still zero, add img token at start and drop one token at end
            if count_img == 0:
                # Insert img_id at pos 0 (replace first token)
                ids_list = input_ids.tolist()
                ids_list[0] = img_id
                input_ids = torch.tensor(ids_list, dtype=torch.long)
                count_img = int((input_ids == img_id).sum().item())

        # Build combined prompt + answer sequence so labels align with logits length
        prompt_len = int(attention_mask.sum().item())
        prompt_ids = input_ids[:prompt_len]

        ans_enc = tok(
            answer,
            truncation=True,
            max_length=self.max_answer_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        answer_ids = ans_enc["input_ids"].squeeze(0)

        # Ensure answer is terminated with eos if available and room remains
        eos_id = tok.eos_token_id
        if eos_id is not None:
            if answer_ids.numel() == 0:
                answer_ids = torch.tensor([eos_id], dtype=torch.long)
            elif answer_ids[-1].item() != eos_id:
                if answer_ids.numel() < self.max_answer_len:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos_id], dtype=torch.long)])
                else:
                    answer_ids[-1] = eos_id

        total_max_len = self.max_prompt_len + self.max_answer_len
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        # Truncate answer if it would exceed allocated space
        max_answer_space = max(0, total_max_len - prompt_len)
        if answer_ids.numel() > max_answer_space:
            answer_ids = answer_ids[:max_answer_space]
        ans_len = answer_ids.numel()

        combined_input_ids = torch.full((total_max_len,), pad_id, dtype=torch.long)
        combined_attention_mask = torch.zeros(total_max_len, dtype=torch.long)
        labels = torch.full((total_max_len,), -100, dtype=torch.long)

        if prompt_len > 0:
            combined_input_ids[:prompt_len] = prompt_ids
            combined_attention_mask[:prompt_len] = 1

        if ans_len > 0:
            combined_input_ids[prompt_len : prompt_len + ans_len] = answer_ids
            combined_attention_mask[prompt_len : prompt_len + ans_len] = 1
            labels[prompt_len : prompt_len + ans_len] = answer_ids

        return {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
