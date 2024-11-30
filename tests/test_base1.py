import pytest

from minbpe.base1 import Tokenizer

def test_load() -> None:
    model_file = 'test.model'
    tokenizer = Tokenizer()
    tokenizer.load(model_file)
    assert tokenizer.pattern == r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    assert tokenizer.special_tokens == {'<|endoftext|>': 100257}
    assert tokenizer.merge == {(97, 97): 256, (101, 97): 257}
    assert tokenizer.vocab[256] == b'aa'
    assert tokenizer.vocab[257] == b'ea'
    

def test_save() -> None:
    model_file = 'test.model'
    tokenizer = Tokenizer()
    tokenizer.load(model_file)
    
    save_prefix = 'test_save'
    tokenizer.save(save_prefix)
    
    