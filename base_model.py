from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/qwen3-8b-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=False,
)

FastLanguageModel.for_inference(model)

prompt = "Hello"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
)