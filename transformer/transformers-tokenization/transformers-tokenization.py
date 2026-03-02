import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        vocab = set()
        for text in texts:
            for word in text.split():
                vocab.add(word)
        
        vocab = list(vocab)
        
        vocab = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ] + sorted(list(vocab))

        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(vocab)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        token_list = []
    
        for word in text.split():
            if word in self.word_to_id:
                token_list.append(self.word_to_id[word])
            else:
                token_list.append(self.word_to_id[self.unk_token])
    
        return token_list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
    
        for token_id in ids:
            words.append(self.id_to_word.get(token_id, self.unk_token))
    
        return " ".join(words)
