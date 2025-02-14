import os
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm  # Progress bar
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

# --------------
# Load Model and Tokenizer
# --------------
model_path = "HuggingFaceTB/SmolLM2-135M"  # Update as needed

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,  # bf16 precision
    attn_implementation="flash_attention_2")

# Ensure the pad token is defined
tokenizer.pad_token = tokenizer.eos_token

# Define dataset path
dataset_path = "/workspace/data/1B.jsonl"
raw_dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# Tokenize dataset
tokenized_dataset = raw_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=["text", "score", "model_output"]
)

tokenized_dataset.set_format(type="torch")

# Create DataLoader
per_device_train_batch_size = 8
train_dataloader = DataLoader(
    tokenized_dataset, 
    batch_size=per_device_train_batch_size, 
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

# Training Setup
num_train_epochs = 1
gradient_accumulation_steps = 1
logging_steps = 100
learning_rate = 5e-5

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.compile(model)

from muon import Muon

muon_params = [p for p in model.parameters() if p.ndim >= 2]
adamw_params = [p for p in model.parameters() if p.ndim < 2]

optimizers = [
    Muon(muon_params, lr=5e-3, momentum=0.95),
    torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.90, 0.95), weight_decay=0.01)
]

num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
if len(train_dataloader) % gradient_accumulation_steps != 0:
    num_update_steps_per_epoch += 1
max_train_steps = num_train_epochs * num_update_steps_per_epoch
max_grad_norm = 1.0

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizers[1],
    num_warmup_steps=max_train_steps // 32,
    num_training_steps=max_train_steps,
)

# Training Loop
global_step = 0
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

model.train()
tokens_counter = 0
total_tokens_counter = 0
logging_start_time = time.time()
loss_log_1k = []
use_autocast = device.type == "cuda" and torch.cuda.is_bf16_supported()
accumulated_loss = 0.0
training_start_time = time.time()

for epoch in range(num_train_epochs):
    print(f"Epoch {epoch + 1}/{num_train_epochs}")
    epoch_iterator = tqdm(train_dataloader, desc="Training", leave=False)
    for optimizer in optimizers:
        optimizer.zero_grad()
    
    for step, batch in enumerate(epoch_iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        tokens_in_batch = batch["input_ids"].numel()
        tokens_counter += tokens_in_batch
        total_tokens_counter += tokens_in_batch
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_autocast else torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        accumulated_loss += loss.item()
        
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            clip_grad_norm_(model.parameters(), max_grad_norm)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1
            
            if global_step % logging_steps == 0:
                current_time = time.time()
                elapsed_time = current_time - logging_start_time
                avg_tokens_per_sec = tokens_counter / elapsed_time if elapsed_time > 0 else 0
                avg_loss = accumulated_loss / logging_steps
                print(f"Step {global_step}: Average Loss (100 steps): {avg_loss:.4f}, Tokens/sec: {avg_tokens_per_sec:.2f}")
                loss_log_1k.append(avg_loss)
                accumulated_loss = 0.0
                tokens_counter = 0
                logging_start_time = current_time

print("Training complete!")
training_end_time = time.time()
total_elapsed_time = training_end_time - training_start_time
overall_tokens_per_sec = total_tokens_counter / total_elapsed_time if total_elapsed_time > 0 else 0
print(f"Overall tokens per second: {overall_tokens_per_sec:.2f}")

# Save Model and Tokenizer
save_dir = "./trained_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved at {save_dir}")
