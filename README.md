# Hyperion-V1

An advanced, high-performance **50M Parameter Language Model** built from scratch in Python and PyTorch. Hyperion-V1 incorporates modern structural enhancements found in production-grade LLMs like LLaMA and Mistral to achieve exceptional training throughput and convergence stability.

##  Key Architectural Features

*   **Grouped-Query Attention (GQA):** Optimizes memory footprint and memory bandwidth usage during decoding by clustering query heads into distinct Key-Value pairs.
*   **FlashAttention-2 Integration:** Utilizes native `torch.nn.functional.scaled_dot_product_attention` fused kernels to maximize GPU hardware compute utilization and accelerate attention calculation.
*   **Deep-Shared Mixture of Experts (MoE):** Combines a high-capacity continuous routing network layer alongside a set of top-k sparse experts using vectorized token routing math to eliminate slow sequential Python loop bottlenecks.
*   **Rotary Position Embeddings (RoPE):** Implements modern relative positional encoding injected directly into query and key channels to enhance structural sequence tracking across a sliding token canvas.
*   **Continuous Token-Streaming Dataset Engine:** Eliminates padding waste by processing text into an unbroken data stream, automatically caching processed tokens into high-speed `.npy` binaries for zero-lag startups.

---

##  Repository Structure

The core files within this repository organize the pipeline as follows:
*   `model.py`: Core network architecture definitions including GQA, FlashAttention wrappers, RoPE tracking layers, and MoE blocks.
*   `tokenizer_utils.py`: Pipeline code to train and initialize a custom sub-word Byte-Pair Encoding (BPE) text tokenizer.
*   `dataset.py`: Performance token dataset layer that maps continuous binary input text streams into tensor segments.
*   `train.py`: The production pre-training loop featuring bfloat16 mixed-precision tracking, gradient accumulation, and custom cosine scheduling.
*   `generate.py`: The autoregressive top-k/top-p inference execution script to sample completion texts.

---

##  Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ and a compatible PyTorch environment established. Install the required Hugging Face text parsing dependencies:

```bash
pip install torch numpy tokenizers
```

### 2. Dataset Preparation
Create a subdirectory named `data` and populate it with your raw unstructured text corpora text files. Ensure you divide your text samples into a training set and validation set:

```bash
mkdir data
# Populate with:
# data/train.txt
# data/val.txt
```

---

##  How To Run

### Step 1: Initialize and Train the Tokenizer
Execute the initialization code in `tokenizer_utils.py` to process your target raw text files and export your structural BPE sub-word map mappings as a `hyperion_tokenizer.json` parameter model asset:

```bash
python tokenizer_utils.py
```

### Step 2: Kick off Pre-Training
Run the primary optimization script. This handles model scaling adjustments, initializes token streams, applies gradient clipping caps (`max_norm=1.0`), manages continuous AMP tensor loops, and exports the top-performing operational matrices to `checkpoints/hyperion_50m_best.pt`:

```bash
python train.py
```

### Step 3: Autoregressive Inference Generation
Once a performance baseline checkpoint has been saved to the checkpoint space, pass context sequences into your generation module to sample predictions using Top-K and Top-P filtering layouts:

```bash
python generate.py
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete details.
