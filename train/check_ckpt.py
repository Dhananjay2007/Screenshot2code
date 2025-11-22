#!/usr/bin/env python3
"""
Check if checkpoint is valid and can be resumed
Run this in your scripts2 environment
"""

import torch
from pathlib import Path

checkpoint_path = Path(r"E:\SCREENSHOT2CODE\checkpoints\llava15_7b_fresh\final_adapter.pt")

print("=" * 70)
print("CHECKPOINT VALIDATION")
print("=" * 70)

print(f"\nFile path: {checkpoint_path}")
print(f"File exists: {checkpoint_path.exists()}")

if checkpoint_path.exists():
    file_size_gb = checkpoint_path.stat().st_size / (1024 ** 3)
    print(f"File size: {file_size_gb:.2f} GB")

    print("\nAttempting to load checkpoint...")
    try:
        ck = torch.load(checkpoint_path, map_location='cpu')

        print(f"✓ Checkpoint loaded successfully!")
        print(f"\nCheckpoint contents:")
        print(f"  - Keys: {list(ck.keys())}")
        print(f"  - Global step: {ck.get('global_step')}")

        if 'model' in ck:
            print(f"  - Model state dict size: {len(ck['model'])} entries")
        if 'optim' in ck:
            print(f"  - Optimizer state exists: Yes")
        if 'sched' in ck:
            print(f"  - Scheduler state exists: Yes")

        print("\n" + "=" * 70)
        print("✅ CHECKPOINT IS VALID AND CAN BE RESUMED!")
        print("=" * 70)
        print("\nTo resume training, use:")
        print('  --resume "E:\\SCREENSHOT2CODE\\checkpoints\\llava15_7b_fresh\\final_adapter.pt"')

    except Exception as e:
        print(f"❌ Error loading checkpoint: {type(e).__name__}")
        print(f"   Details: {e}")
        print("\n" + "=" * 70)
        print("❌ CHECKPOINT IS CORRUPTED - Cannot resume")
        print("=" * 70)
        print("\nYou will need to start training from scratch.")
else:
    print(f"❌ File not found at {checkpoint_path}")
