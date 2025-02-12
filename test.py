import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load model and tokenizer (use NanoGPT, Mistral, or Llama2)
model_name = "gpt2"  # Change to your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

print(model)
# Sample input prompt
input_text = "The future of AI systems programming is"
inputs = tokenizer(input_text, return_tensors="pt")

# Measure inference speed
num_tokens = 128  # Generate 128 new tokens
start_time = time.time()
output = model.generate(**inputs, max_length=num_tokens)
end_time = time.time()

# Calculate tokens per second
tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]
tps = tokens_generated / (end_time - start_time)

print(f"Inference Speed: {tps:.2f} tokens/sec")
