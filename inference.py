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
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# ==========================
# RUN INFERENCE
# ==========================
prompt = (
    "<|im_start|>system\nYou are a polite telecommunications customer support assistant. Only answer the user's explicit question. Do not assume or jump to technical troubleshooting unless specifically asked.<|im_end|>\n"
    "<|im_start|>user\nWhat is the purpose of the Nmfaf_3daDataManagement_Deconfigure service operation?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))