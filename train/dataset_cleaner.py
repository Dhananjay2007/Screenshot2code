#!/usr/bin/env python3
"""
Utility to validate and clean your training dataset.
Removes entries with missing images and generates a cleaned JSON.
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_image_path(images_root: Path, rel: str) -> Path:
    """Resolve image path using common patterns."""
    root = images_root
    rel = rel.replace("\\", "/")

    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p

    if root.name.lower() == "images" and rel.lower().startswith("images/"):
        rel = rel[len("images/"):]
        return root / rel
    if rel.lower().startswith("images/"):
        return root / rel[len("images/"):]

    candidate = root / rel
    if candidate.exists():
        return candidate

    candidate2 = root / "images" / rel
    if candidate2.exists():
        return candidate2

    return root / rel


def clean_dataset(json_path: str, images_root: str, output_path: str = None):
    """
    Validate dataset and remove entries with missing images.

    Args:
        json_path: Path to JSON file
        images_root: Root directory for images
        output_path: Path to save cleaned JSON (default: append _cleaned)
    """
    json_path = Path(json_path)
    images_root = Path(images_root)

    if output_path is None:
        output_path = json_path.with_stem(json_path.stem + "_cleaned")

    logger.info(f"Loading dataset from {json_path}")
    items: List[Dict] = json.loads(json_path.read_text(encoding="utf-8"))

    logger.info(f"Total items: {len(items)}")

    cleaned_items = []
    missing_count = 0

    for idx, item in enumerate(items):
        img_path = resolve_image_path(images_root, item["image"])

        if img_path.exists():
            cleaned_items.append(item)
        else:
            logger.warning(f"Missing image for item {idx}: {img_path}")
            missing_count += 1

    logger.info(
        f"Cleaned dataset: {len(cleaned_items)} valid items "
        f"({missing_count} removed)"
    )

    output_path.write_text(json.dumps(cleaned_items, indent=2), encoding="utf-8")
    logger.info(f"Saved cleaned dataset to {output_path}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean dataset by removing missing images")
    ap.add_argument("--json", required=True, help="Path to JSON dataset file")
    ap.add_argument("--images_root", required=True, help="Root directory for images")
    ap.add_argument("--output", default=None, help="Output path (default: auto)")
    args = ap.parse_args()

    clean_dataset(args.json, args.images_root, args.output)
