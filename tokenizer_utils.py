import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(corpus_path, vocab_size=4000, save_path="hyperion_tokenizer.json"):
    """
    Trains a custom Byte-Pair Encoding (BPE) tokenizer on a text corpus file.
    
    Args:
        corpus_path (str): Path to the raw .txt file containing training text.
        vocab_size (int): Size of the token vocabulary.
        save_path (str): File path where the trained tokenizer configuration will be saved.
    """
    print(f"--- Training BPE Tokenizer (Vocab Size: {vocab_size}) ---")
    
    # Initialize a clean BPE model configuration with an Unknown Token fallback
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Split text on whitespace to prevent tokens from crossing word boundaries
    tokenizer.pre_tokenizer = Whitespace()
    
    # Define standard special tokens used to handle framing and boundary sequences
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], 
        vocab_size=vocab_size
    )
    
    # Train and save the output configuration asset
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer trained and successfully saved to: {save_path}\n")
    return tokenizer

def load_tokenizer(tokenizer_path="hyperion_tokenizer.json"):
    """Loads a pre-trained tokenizer configuration asset from disk."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file '{tokenizer_path}' not found. "
            f"Please run 'train_bpe_tokenizer' or generate it first."
        )
    return Tokenizer.from_file(tokenizer_path)

if __name__ == "__main__":
    # Create a small sample corpus to run a file structural dry run
    sample_corpus = "sample_corpus.txt"
    with open(sample_corpus, "w", encoding="utf-8") as f:
        f.write("Deep learning and transformer architectures have completely revolutionized modern AI.\n")
        f.write("Mixture of Experts scales neural networks efficiently.\n")

    # Train structural validation test
    tok = train_bpe_tokenizer(sample_corpus, vocab_size=1000)
    
    # Test encoding and decoding stability
    text = "Mixture of Experts architectures."
    encoded = tok.encode(text)
    decoded = tok.decode(encoded.ids)
    
    print("--- Tokenizer Functional Validation Check ---")
    print(f"Original Text: {text}")
    print(f"Encoded Token IDs: {encoded.ids}")
    print(f"Decoded Clean Text: {decoded}")
    
    # Clean up temporary structural file
    if os.path.exists(sample_corpus):
        os.remove(sample_corpus)
