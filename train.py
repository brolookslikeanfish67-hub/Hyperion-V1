import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project dependency imports
from model import HyperionV1, VOCAB_SIZE, DIM, MAX_SEQ_LEN
from tokenizer_utils import load_tokenizer
from dataset import HyperionDataset

# ==========================================
# 1. TRAINING CONFIGURATION & HYPERPARAMETERS
# ==========================================
# File system directory structures
DATA_DIR = "data"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
TOKENIZER_JSON = "hyperion_tokenizer.json"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "hyperion_50m_best.pt")

# Optimization execution parameters
BATCH_SIZE = 32          # Micro-batch size per optimization step
GRAD_ACCUM_STEPS = 4     # Simulates larger global batch sizes via gradient aggregation
MAX_STEPS = 5000         # Total training step limit
VAL_INTERVAL = 250       # Validation evaluation frequency bound
SAVE_INTERVAL = 500      # Periodic backup checkpoint frequency

# Optimizer settings (Decoupled weight decay parameters)
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Hardware runtime target resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Initialization: Targeting device backend {DEVICE.upper()} ---")

# Guarantee required output subdirectories are provisioned
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 2. LEARNING RATE SCHEDULE (COSINE WITH WARMUP)
# ==========================================
def get_learning_rate(step, max_steps, warmup_steps, base_lr, min_lr):
    """
    Computes a learning rate multiplier adhering to a cosine decay schedule 
    preceded by a linear warmup ramp phase.
    """
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step > max_steps:
        return min_lr
    
    # Mathematical implementation of standard cosine decay transformation
    decay_ratio = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)

# ==========================================
# 3. EVALUATION FUNCTION MODULE
# ==========================================
@torch.no_grad()
def evaluate_loss(model, dataloader, device, eval_iters=50):
    """
    Computes a loss baseline metric aggregated across a target validation 
    dataset segment while operating under inference constraints.
    """
    model.eval()
    total_loss = 0.0
    actual_iters = min(len(dataloader), eval_iters)
    
    device_type = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    for idx, (x, y) in enumerate(dataloader):
        if idx >= actual_iters:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
            _, loss = model(x, targets=y)
        total_loss += loss.item()
        
    model.train()
    return total_loss / max(1, actual_iters)

# ==========================================
# 4. TRAINING EXECUTION PIPELINE
# ==========================================
def main():
    # Verify core system pre-conditions are satisfied before optimization loop
    if not os.path.exists(TOKENIZER_JSON) or not os.path.exists(TRAIN_TXT):
        print(f"Configuration Error: Ensure target corpus paths and '{TOKENIZER_JSON}' exist.")
        return

    # Initialize vocabulary assets
    tokenizer = load_tokenizer(TOKENIZER_JSON)

    # Initialize dataset streaming loaders
    print("Configuring tokenized input streaming engines...")
    train_dataset = HyperionDataset(TRAIN_TXT, tokenizer, max_seq_len=MAX_SEQ_LEN)
    val_dataset = HyperionDataset(VAL_TXT, tokenizer, max_seq_len=MAX_SEQ_LEN) if os.path.exists(VAL_TXT) else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, pin_memory=True) if val_dataset else None

    # Instantiate model backbone
    model = HyperionV1().to(DEVICE)
    print(f"Model Architecture Compiled: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M Parameters.")

    # Configure high-throughput optimization parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.9, 0.95), 
        weight_decay=WEIGHT_DECAY,
        fused=True if DEVICE == "cuda" else False
    )

    # Run state tracking initialization
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    
    device_type = "cuda" if DEVICE == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print("\nStarting Hyperion-V1 pre-training execution pipeline...")
    model.train()

    # Core execution block
    while step < MAX_STEPS:
        for x_batch, y_batch in train_loader:
            if step >= MAX_STEPS:
                break
                
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            # Update schedule position step metrics
            current_lr = get_learning_rate(step, MAX_STEPS, WARMUP_STEPS, LEARNING_RATE, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Autocast accelerated matrix compute loop
            with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                logits, loss = model(x_batch, targets=y_batch)
                loss = loss / GRAD_ACCUM_STEPS

            # Compute backpropagation values
            loss.backward()

            # Execute gradient application thresholds once accumulation step criteria are met
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Periodic Validation Checkpoint Logic
            if step % VAL_INTERVAL == 0 and step > 0:
                if val_loader:
                    val_loss = evaluate_loss(model, val_loader, DEVICE)
                    print(f"Step: {step:05d} | Combined Train Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | Validation Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
                    
                    # Serialize optimal state profiles when breaking historical validation minimas
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                        }, BEST_MODEL_PATH)
                        print(f"Saved optimized parameter snapshot artifact to: {BEST_MODEL_PATH}")
                else:
                    print(f"Step: {step:05d} | Combined Train Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | LR: {current_lr:.2e}")

            # Backup Save State Engine 
            if step % SAVE_INTERVAL == 0 and step > 0:
                periodic_path = os.path.join(CHECKPOINT_DIR, f"hyperion_step_{step}.pt")
                torch.save({'model_state_dict': model.state_dict()}, periodic_path)

            step += 1

    total_duration = time.time() - start_time
    print(f"\nTraining pipeline concluded successfully in {total_duration/60:.2f} minutes.")

if __name__ == "__main__":
    main()
