import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HyperionDataset(Dataset):
    """
    Token-Streaming Dataset.
    Flattens the entire corpus into a continuous stream to completely eliminate 
    wasted padding tokens and maximize GPU training throughput.
    """
    def __init__(self, corpus_path, tokenizer, max_seq_len=1024, force_rebuild=False):
        """
        Args:
            corpus_path (str): Path to the raw text file to process.
            tokenizer (Tokenizer): Pre-trained BPE tokenizer instance from tokenizer_utils.py.
            max_seq_len (int): Maximum sequence token window length (MAX_SEQ_LEN).
            force_rebuild (bool): If True, ignores existing token caches and rebuilds from scratch.
        """
        self.max_seq_len = max_seq_len
        cache_path = corpus_path + ".tokens.npy"
        
        # 1. Fast Cache Loading
        if os.path.exists(cache_path) and not force_rebuild:
            print(f"--- Loading Tokenized Dataset Cache from: {cache_path} ---")
            self.tokens = np.load(cache_path)
        else:
            print(f"--- Processing Dataset (Continuous Streaming Pass): {corpus_path} ---")
            all_token_ids = []
            
            # Read text in chunks to prevent memory blowouts on huge files
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Tokenize and append directly to the global stream
                    all_token_ids.extend(tokenizer.encode(line).ids)
            
            self.tokens = np.array(all_token_ids, dtype=np.uint16)
            print(f"Saving compiled binary token cache to: {cache_path}")
            np.save(cache_path, self.tokens)

        # Calculate exact number of training sequences we can extract
        # We need (max_seq_len + 1) tokens per slice to get input X and target Y
        self.num_sequences = (len(self.tokens) - 1) // self.max_seq_len
        print(f"Dataset compiled. Total real tokens: {len(self.tokens):,}")
        print(f"Total zero-padding training sequences: {self.num_sequences:,}\n")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Calculate sliding target index boundaries
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len + 1
        
        # Pull sequence slice
        chunk = torch.from_numpy(self.tokens[start_idx:end_idx].astype(np.int64))
        
        # x: Input sequence (tokens 0 to N-1)
        # y: Target sequence (tokens 1 to N, shifted by 1 position for next-token prediction)
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y

if __name__ == "__main__":
    from tokenizer_utils import train_bpe_tokenizer
    
    # Structural verification test code block
    temp_text = "test_corpus.txt"
    with open(temp_text, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(f"Line {i}: This is a high throughput production ready transformer data stream layer validation check.\n")

    # Train a minimal validator configuration instance
    tokenizer = train_bpe_tokenizer(temp_text, vocab_size=100)
    
    # Initialize the high performance dataset wrapper
    dataset = HyperionDataset(temp_text, tokenizer, max_seq_len=16, force_rebuild=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Extract one sample batch to verify logic maps cleanly
    x_sample, y_sample = next(iter(dataloader))
    
    print("--- Dataset Functional Verification Check ---")
    print(f"Input Batch Shape (X):  {x_sample.shape} (Expected: [Batch Size, MAX_SEQ_LEN])")
    print(f"Target Batch Shape (Y): {y_sample.shape} (Expected: [Batch Size, MAX_SEQ_LEN])")
    print(f"Sample X Token IDs: {x_sample[0].tolist()}")
    print(f"Sample Y Token IDs: {y_sample[0].tolist()} (Shifted by 1)")
    
    # Clean up temporary mock assets
    if os.path.exists(temp_text):
        os.remove(temp_text)
    if os.path.exists(temp_text + ".tokens.npy"):
        os.remove(temp_text + ".tokens.npy")
    if os.path.exists("hyperion_tokenizer.json"):
        os.remove("hyperion_tokenizer.json")
