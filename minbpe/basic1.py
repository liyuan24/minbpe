"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from copy import copy
from minbpe.base1 import Tokenizer
from minbpe.utilities import get_stats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        
    def train(self, text: str, vocab_size: int, verbose: bool=False):
        assert vocab_size >= 256
        # convert to bytes
        text_bytes = text.encode('utf-8')
        # convert bytes to integers
        inds = list(text_bytes)
        num_merges = vocab_size - 256
        idx = 256
        merges = {}
        vocab = {ind: bytes([ind]) for ind in range(256)}
        for i in range(num_merges):
            # find the most frequent pair
            stats = get_stats(inds)
            top_pair = max(stats, key=stats.get)
            inds = merge(inds, top_pair, idx)
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx}, {top_pair} has {stats[top_pair]} occurences")    
            idx += 1
        self.merge = merges # used in encode
        self.vocab = vocab # used in decode
    
    def encode(self, text: str) -> list[int]:
        byte_text = text.encode('utf-8')
        inds = list(byte_text)
        # use merge to merge tokens
        while len(inds) > 2:
            stats = get_stats(inds)
            # get the pair with lowest merge index in self.merge
            # if no pair in self.merge, the ranking if positive infinity to make sure min works
            pair = min(stats, key=lambda p: self.merge.get(p, float('inf')))
            if pair not in self.merge:
                # no more merges could be made
                break
            inds = merge(inds, pair, self.merge[pair])
        return inds

    def decode(self, inds: list[int]) -> str:
        byte_str = b''.join(self.vocab[ind] for ind in inds)
        return byte_str.decode('utf-8', errors='replace')
        
                    
            
            
            
            
        
    