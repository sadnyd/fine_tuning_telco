from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# ==========================
# CONFIG
# ==========================
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER = "./telco-qwen/final" 
# ==========================
# LOAD MODEL
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

model.eval()
prompt = "Hello"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
)

print(tokenizer.decode(outputs[0]))