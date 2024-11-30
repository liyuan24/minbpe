import pytest
from minbpe.utilities import get_stats, merge, replace_control_characters, render_tokens

def test_get_stats():
    ids = [1, 2, 3, 1, 2, 4]
    res = get_stats(ids)

    assert res == {(1, 2): 2, (2, 3): 1, (3, 1): 1, (2, 4): 1}

@pytest.mark.parametrize("ids, pair, idx, res", [([1, 2, 3, 1, 2, 4], (2, 3), 5, [1, 5, 1, 2, 4]), ([1, 1, 1, 1, 1], (1, 1), 2, [2, 2, 1])])
def test_merge(ids, pair, idx, res):
    print(f"ids: {ids}")
    print(f"pair: {pair}")
    print(f"idx: {idx}")
    print(f"res: {res}")
    test_res = merge(ids, pair, idx)
    assert test_res == res

@pytest.mark.parametrize("s, expected_res", [("abc\nd", "abc\\u000ad"), ("eee\rabd", "eee\\u000dabd")])
def test_replace_control_characters(s, expected_res):
    # if we don't escape for example \\u000a, \u000a is actually \n, so we escape 
    # to print its code point to not distort the print by control characters
    res = replace_control_characters(s)
    assert res == expected_res


@pytest.mark.parametrize("byte_tokens, expected_res", [(b"i love you, xinran", "i love you, xinran"), (b"this is a new line, \n, haha", "this is a new line, \\u000a, haha")])
def test_render_tokens(byte_tokens, expected_res):
    # if you print "this is a new line, \\u000a, haha", it will be like `this is a new line, \u000a, haha`
    res = render_tokens(byte_tokens)
    assert res == expected_res

    
    


