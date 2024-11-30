from minbpe.basic1 import BasicTokenizer


def test_train() -> None:
    tokenizer = BasicTokenizer()
    text = "abacbadbad"
    vocab_size = 258
    tokenizer.train(text, vocab_size)
    assert tokenizer.merge == {(98, 97): 256, (256, 100): 257}
    assert tokenizer.vocab[256] == b'ba'
    assert tokenizer.vocab[257] == b'bad'


def test_encode() -> None:
    tokenizer = BasicTokenizer()
    text = "abacbadbad"
    vocab_size = 258
    tokenizer.train(text, vocab_size)
    res = tokenizer.encode(text)
    assert res == [97, 256, 99, 257, 257]
    

def test_decode() -> None:
    tokenizer = BasicTokenizer()
    text = "abacbadbad"
    vocab_size = 258
    tokenizer.train(text, vocab_size)
    
    tokens = [97, 256, 99, 257, 257]
    res = tokenizer.decode(tokens)
    assert res == "abacbadbad"
    