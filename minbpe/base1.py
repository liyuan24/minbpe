# the base Tokenizer class

'''
the Tokenizer has two jobs
1. encode a string/text into tokens(integers)
2. decode token integers into string/text

So 
1. merge: dict[(int, int), int] is recording which pair of tokens are merged together into a new token.
   The merge will help reduce the length of tokens for a given input string/text. Why we need this reduction is
   that it will help reduce the length of context and the benefit is that in Transformer model, each token
   better attend to each other given the limited context length
2. vocab: dict[int, bytes] or dict[bytes, int]. This will tell us given a list of integer tokens, what are their
   corresponding bytes. It will be used in decode process.

vocab is from 3 places
1. for unicode code points from 0 to 255, they have a 1:1 mapping between integer and bytes
2. merge: it will tell us the merged tokens and their parents. We build the bytes recursively
3. special_tokens: those are tokens used for special purposes 
'''
from .utilities import render_tokens


class Tokenizer:
    def __init__(self):
        self.merge = {} # (int, int) -> int
        # pattern is regular expression pattern to divide the groups of merges
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g, {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
    
    
    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # build from merge, the order matters
        for (id1, id2), id3 in self.merge.items():
            # please note that two bytes concatenated toegether might not be a valid UTF-8 bytes
            vocab[id3] = vocab[id1] + vocab[id2]
        
        # build from special tokens
        for s, idx in self.special_tokens.items():
            vocab[idx] = s.encode('utf-8')
        return vocab

    def train(self, text: str, vocab_size: int, verbose: bool=False):
        # Tokenizer will train a vocab of size vocab_size from text
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        # encode the given text string
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        # decode a list of tokens into a string text
        raise NotImplementedError
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired by sentencepiece's model saving:
        - model file is the critical one for load()
        - vocab file is just a pretty printed version for human inspection
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that is all we need
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special_token, idx in self.special_tokens.items():
                f.write(f"{special_token} {idx}\n")
            # the merge dict, only key pair, the value will be derived from 256 onward when loading
            for idx1, idx2 in self.merge:
                f.write(f"{idx1} {idx2}\n")
        vocab_file = file_prefix + ".vocab"
        # it will be derived from vocab(int -> bytes)
        # and we will also get its parents from the inverted merge dict
        # the format of the file
        # [p1_str, p2_str] -> [child_str] child_token_integer
        inverted_merge = {id: pair for pair, id in self.merge.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for id, token in self.vocab.items():
                # many byte tokens might be partial utf-8 sequences
                # and cannot be decoded into a valid string. Here we are using errors='replace' to
                # replace them with char �. That also means that we cannot possibly use .vocab in load()
                # because decoding in this way is a lossy operation
                s = render_tokens(token)
                if id in inverted_merge:
                    p1, p2 = inverted_merge[id]
                    s1, s2 = render_tokens(self.vocab[p1]), render_tokens(self.vocab[p2])
                    f.write(f"[{s1}][{s2}] -> [{s}] {id}\n")
                else:
                    # otherwise this is a leaf token, just print it
                    f.write(f"[{s}] {id}\n")
    
    def load(self, model_file: str) -> None:
        """Inverse of save() but only for model file"""
        assert model_file.endswith('.model')
        merge = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding='utf-8') as f:
            # read version
            version = f.readline().strip()
            assert version == 'minbpe v1'
            # read pattern
            self.pattern = f.readline().strip()
            # read special tokens
            num_special_tokens = int(f.readline().strip())
            # read each special token
            for i in range(num_special_tokens):
                special_token, special_token_idx = f.readline().strip().split()
                special_tokens[special_token] = int(special_token_idx)
            # read merges
            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merge[(idx1, idx2)] = idx
                idx += 1
        
        self.merge = merge
        print(self.merge)
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
        
                
                
                


