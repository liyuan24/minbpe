"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base1 import Tokenizer
from .utilities import get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
'''
1. '(?:[sdmt]|ll|ve|re)
   ?: is non capturing group where matched group can not be referenced later to improve the efficiency of regular expression engine
   [sdmt] is a character class which will match any character in []
   | is OR
   So this will match any s,d,m,t or ll or ve or re
2.  ?\p{L}+: this will match optional empty space followed by one or more letters
3.  ?\p{N}+: this will match optional empty space followed by one or more numbers
4.  ?[^\s\p{L}\p{N}]+: optional empty space followed by characters that are NOT white spaces, or letters or numbers. Basically this will match punctuations
    \s will match white spaces including
        Space: ' ' (space character)
        Tab: '\t' (tab character)
        Newline: '\n' (line feed)
        Carriage return: '\r'
        Form feed: '\f'
        Vertical tab: '\v'
5.\s+(?!\S): one or more white spaces but only matched until it is not followed by a non white space character
   (?!pattern) is negative lookahead which will assert that what follows the current position of the string is NOT matched by pattern
    It prevents matching whitespace if it is followed immediately by a non-whitespace character.
6.\s+: this is a fall back, which will match whitespaces
'''
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
'''
1. '(?i:[sdmt]|ll|ve|re): this is similar to above, and i means case insensitive
2. [^\r\n\p{L}\p{N}]?+\p{L}+: 
   a. [^]: negation
   b. ?+ optional and greedily match. ? will only match one if exists, but ?+ will match as many as possible
3. \p{N}{1,3}: match numbers defined by unicode, at least one but no more than 3
4.  ?[^\s\p{L}\p{N}]++[\r\n]*: 
  a. optional empty space
  b.followed by non white spaces, or letter or number
  c. ++ means one or more but doesn't allow backtracking
  d. zero or more \r or \n
5. same as above
6. same as above 
'''
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # compiled the pattern so that it would be recompiled when used
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.reverse_special_tokens = {}
    
    '''
    we need to consider splitting the training text into different groups via regex pattern.
    The merge can only happen within each group.
    The top pair is the top pair across different groups
    '''
    def train(self, text: str, vocab_size: int, verbose:bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # divide the training text into different groups using pattern
        text_chunks = self.compiled_pattern.findall(text)
        utf8_chunks = [chunk.encode('utf-8') for chunk in text_chunks]
        chunk_inds = [list(chunk) for chunk in utf8_chunks]
        self.vocab = {ind: bytes([ind]) for ind in range(256)}
        merge_idx = 256
        for i in range(num_merges):
            stats = {}
            for chunk in chunk_inds:
                # pass stats so that it will be updated for each chunk
                get_stats(chunk, counts=stats)
            top_pair = max(stats, key=stats.get)
            self.merge[top_pair] = merge_idx
            self.vocab[merge_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            # merge for each chunk
            chunk_inds = [merge(chunk, top_pair, merge_idx) for chunk in chunk_inds]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {merge_idx}, {top_pair} has {stats[top_pair]} occurences") 
            merge_idx += 1
    
    
    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = special_tokens
        self.register_special_tokens = {v:k for k, v in special_tokens.items()}
    
    '''
    need to handle special tokens
    '''
    def decode(self, inds: list[int]) -> str:
        print(f"haha: vocab: {self.vocab}")
        byte_tokens = []
        for ind in inds:
            if ind in self.special_tokens:
                byte_tokens.append(self.register_special_tokens[ind].encode('utf-8'))
            elif ind in self.vocab:
                byte_tokens.append(self.vocab[ind])
            else:
                raise ValueError(f"invalid token id: {ind}")
        byte_str = b''.join(byte_tokens)
        return byte_str.decode('utf-8', errors='replace')

    '''
    1. need to handle special tokens
    2. need to first split the text into groups via pattern
    '''
    def encode(self, text: str, allowed_special="none_raise") -> list[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        specials = None
        if allowed_special == 'all':
            specials = self.special_tokens
        elif allowed_special == 'none':
            specials = {}
        elif allowed_special == 'none_raise':
            specials = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            specials = {k:v for k, v in self.special_tokens.items if k in allowed_special}
        else:
            raise ValueError(f"allowed_special {allowed_special} is not understood")
        
        if not specials:
            return self.encode_ordinary(text)

        # split the text by special tokens, use captured group so that re.split also return the special tokens
        # since the special tokens might include special characters used in regex, we need to escape them
        special_pattern = "(" + "|".join([re.escape(special_token) for special_token in specials]) + ")"
        text_chunks = re.spit(special_pattern, text)
        ids = []
        for text_chunk in text_chunks:
            if text_chunk in specials:
                ids.append(specials[text_chunk])
            else:
                ids.extend(self.encode_ordinary(text_chunk))
        return ids

    def encode_ordinary(self, text: str):
        """
        split the text by patterns first
        """
        text_chunks = self.compiled_pattern.findall(text)
        ids = []
        for chunk in text_chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, text: str):
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
        
        
        
