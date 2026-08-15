import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperion-V1 Config  ---
VOCAB_SIZE = 4000
DIM = 512
N_LAYERS = 6
N_HEADS = 8
N_KV_HEADS = 2           # Grouped-Query Attention (GQA)
N_EXPERTS = 8            # MoE Capacity
TOP_K = 2                # Top-K Routing
WINDOW_SIZE = 128        # Sliding Window Attention (SWA)
MAX_SEQ_LEN = 1024

# Precision dimensions to hit the 50M parameter footprint
EXPERT_HIDDEN_DIM = 480   
SHARED_HIDDEN_DIM = 960  

class RMSNorm(nn.Module):
    """Production-Grade Robust RMSNorm to safeguard against numeric instabilities."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def apply_rope(x, cos, sin):
    """Applies Rotary Position Embeddings (RoPE) to a tensor."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class RotaryEmbedding(nn.Module):
    """Precomputes sinusoidal frequencies for RoPE."""
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

class HyperionAttention(nn.Module):
    """Ultra-Fast Attention using native FlashAttention and GQA."""
    def __init__(self, is_sliding_window=True):
        super().__init__()
        self.is_sliding_window = is_sliding_window
        self.n_heads = N_HEADS
        self.n_kv_heads = N_KV_HEADS
        self.head_dim = DIM // N_HEADS
        self.num_queries_per_kv = N_HEADS // N_KV_HEADS

        # Fused QKV Projection for maximum GPU execution throughput
        self.qkv_proj = nn.Linear(DIM, (N_HEADS + 2 * N_KV_HEADS) * self.head_dim, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        
        # Project and slice Q, K, V in one single execution pass
        qkv = self.qkv_proj(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embeddings
        q = apply_rope(q, rope_cos[:T], rope_sin[:T]).transpose(1, 2)
        k = apply_rope(k, rope_cos[:T], rope_sin[:T]).transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads for GQA compatibility
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Construct efficient training attention masks
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.is_sliding_window:
            window_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-WINDOW_SIZE + 1)
            mask = mask & window_mask
        
        # Leverage FlashAttention-2 pathways natively available in PyTorch
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask.unsqueeze(0).unsqueeze(1), 
            dropout_p=0.0, 
            is_causal=False
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class SwiGLUExpert(nn.Module):
    """A single high-performance SwiGLU expert feed-forward network."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(DIM, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, DIM, bias=False)
        self.w3 = nn.Linear(DIM, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class HyperionMoE(nn.Module):
    """Deep-Shared Mixture of Experts with parallelized vector dispatch and load balancing."""
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(DIM, N_EXPERTS, bias=False)
        self.routed_experts = nn.ModuleList([SwiGLUExpert(EXPERT_HIDDEN_DIM) for _ in range(N_EXPERTS)])
        self.shared_expert = SwiGLUExpert(SHARED_HIDDEN_DIM)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # 1. Evaluate shared core
        shared_out = self.shared_expert(x_flat)
        
        # 2. Gate calculations
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        topk_weights, topk_indices = torch.topk(routing_weights, TOP_K, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Vectorized MoE dispatch pattern (Removes slow Python for-loops over tokens)
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            token_idx, topk_pos = torch.where(topk_indices == i)
            if token_idx.numel() > 0:
                expert_output = expert(x_flat[token_idx])
                weights = topk_weights[token_idx, topk_pos].unsqueeze(-1)
                routed_out.index_add_(0, token_idx, expert_output * weights)

        # Switch Transformer style load-balancing penalty formulation
        p_i = routing_weights.mean(dim=0)
        f_i = torch.bincount(topk_indices.view(-1), minlength=N_EXPERTS).float() / (topk_indices.numel() + 1e-6)
        balance_loss = N_EXPERTS * torch.sum(p_i * f_i)

        final_output = (0.5 * shared_out + 0.5 * routed_out).view(B, T, C)
        return final_output, balance_loss

class HyperionBlock(nn.Module):
    """Alternates between Global and Sliding Window Attention alongside MoE."""
    def __init__(self, layer_idx):
        super().__init__()
        is_swa = (layer_idx % 2) == 0  
        self.attn = HyperionAttention(is_sliding_window=is_swa)
        self.moe = HyperionMoE()
        self.norm1 = RMSNorm(DIM)
        self.norm2 = RMSNorm(DIM)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        moe_out, layer_balance_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, layer_balance_loss

class HyperionV1(nn.Module):
    """The complete Hyperion-V1 Language Model Transformer network."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.rope = RotaryEmbedding(DIM // N_HEADS)
        self.blocks = nn.ModuleList([HyperionBlock(i) for i in range(N_LAYERS)])
        self.norm_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        
        # Tie weights to save VRAM memory footprints
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        rope_cos, rope_sin = self.rope.cos[:T], self.rope.sin[:T]
        
        aux_loss = 0.0
        for block in self.blocks:
            x, layer_balance_loss = block(x, rope_cos, rope_sin)
            aux_loss += layer_balance_loss

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss += 0.01 * aux_loss 

        return logits, loss

if __name__ == "__main__":
    # Dry-run structure test
    model = HyperionV1()
    dummy_inputs = torch.randint(0, VOCAB_SIZE, (2, 256))
    logits, _ = model(dummy_inputs)
    print("--- Hyperion-V1 Architecture File Loaded ---")
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Sanity Check Logits Shape: {logits.shape}")
