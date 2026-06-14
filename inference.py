from unsloth import FastLanguageModel
import torch

# Load the saved model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./telco-qwen/final",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Enable native 2x faster inference (disabling as Unsloth custom kernels can cause ROCm hardware crashes)
# FastLanguageModel.for_inference(model)

# The prompt format based on the training output
prompt_format = """<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"""

def generate_response(instruction):
    inputs = tokenizer(
        [prompt_format.format(instruction)], return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 256,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id,
    )
    
    # Decode the response
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("\n--- Model Output ---")
    print(decoded_output)
    print("--------------------\n")

if __name__ == "__main__":
    print("Model loaded successfully. Ready for inference!")
    while True:
        user_input = input("Enter a telco query (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        generate_response(user_input)
