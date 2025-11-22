import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
OUTPUT = "E:/SCREENSHOT2CODE/checkpoints/manual_save.pt"

print("Loading 4bit base model…")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
)

print("Attaching current LoRA…")
model = PeftModel.from_pretrained(model, "E:/SCREENSHOT2CODE/checkpoints/llava15_7b_fresh")

print("Extracting only the LoRA adapter weights…")
state = model.state_dict()

print("Saving…")
torch.save({"model": state}, OUTPUT)

print(f"✔ Saved current training state to: {OUTPUT}")
