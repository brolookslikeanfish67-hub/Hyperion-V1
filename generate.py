import os
import torch
import torch.nn.functional as F

# Project dependency imports
from model import HyperionV1, VOCAB_SIZE, DIM
from tokenizer_utils import load_tokenizer

# ==========================================
# 1. INFERENCE RUNTIME CONFIGURATION
# ==========================================
CHECKPOINT_PATH = os.path.join("checkpoints", "hyperion_50m_best.pt")
TOKENIZER_JSON = "hyperion_tokenizer.json"

# Hardware target optimization resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. SAMPLING ENGINE IMPLEMENTATION
# ==========================================
def sample_top_k_top_p(logits, temperature=0.7, top_k=50, top_p=0.9):
    """
    Applies Temperature, Top-K scaling, and Top-p (Nucleus) filtering 
    to raw prediction logits to select the next token index.
    """
    # 1. Apply generation temperature scaling
    logits = logits / max(temperature, 1e-5)
    
    # 2. Apply Top-K filtering bounds
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
    # 3. Apply Top-p (Nucleus) filtering bounds
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Shift mask right to keep the first token that exceeds top_p thresholds
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
    # Convert filtered distribution vectors into precise token IDs
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# ==========================================
# 3. AUTOREGRESSIVE GENERATION INTERACTION
# ==========================================
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """
    Sequentially appends model generation passes onto a base prompt string sequence.
    """
    model.eval()
    
    # Encode prompt string directly to input tensor
    encoded_prompt = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=DEVICE)
    
    # Unpack boundaries to handle generation loops cleanly
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Enforce maximum text context width tracking
        context_ids = input_ids[:, -1024:]
        
        # Execute forward pass through structural network blocks
        logits, _ = model(context_ids)
        
        # Isolate last logit step metrics mapping
        next_token_logits = logits[0, -1, :]
        
        # Process distributions through filtering configurations
        next_token = sample_top_k_top_p(
            next_token_logits, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        
        # Append token directly back to internal memory traces
        generated_tokens.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        
        # Terminate token iteration early if End-Of-Sequence markers are hit
        if next_token.item() == tokenizer.token_to_id("[EOS]"):
            break
            
    # Convert token arrays clean back to structured text strings
    decoded_completion = tokenizer.decode(generated_tokens)
    return decoded_completion

# ==========================================
# 4. EXECUTION ENTRANCE MAIN SCRIPT
# ==========================================
def main():
    # Verify pre-conditions are satisfied before initiating runtime procedures
    if not os.path.exists(TOKENIZER_JSON):
        print(f"Error: Target tokenization config mapping asset '{TOKENIZER_JSON}' not found.")
        return

    tokenizer = load_tokenizer(TOKENIZER_JSON)
    model = HyperionV1().to(DEVICE)
    
    # Load model state snapshot dictionary or skip with warning alerts
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint parameters from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded successfully. System online.\n")
    else:
        print(f" Warning: Checkpoint target file '{CHECKPOINT_PATH}' not found.")
        print("Operating under un-trained raw initialization matrices baseline profiles.\n")

    # Sample interactive text completion demonstration pass
    sample_prompt = "Deep learning and transformer architectures"
    print(f"Prompt Input Sequence: '{sample_prompt}'")
    print("Generating completion sequence...")
    
    completion = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=sample_prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40,
        top_p=0.85
    )
    
    print(f"\nGenerated Output:\n{sample_prompt} {completion}")

if __name__ == "__main__":
    main()
