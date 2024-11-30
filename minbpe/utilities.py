import unicodedata

def get_stats(ids: list[int], counts:dict=None):
    """
    Given a list of integers, return a dictionary of counts consecutive pairs of integers
    Example: [1, 2, 3, 1, 2, 4] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1, (2, 4): 1}
    Optionally update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for id1, id2 in zip(ids[:-1], ids[1:]):
        counts[(id1, id2)] = counts.get((id1, id2), 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int], idx: int):
    """
    Given a list of integers, replace the consecutive occurences of pair with new integer idx
    Example: ids=[1, 2, 3, 1, 2, 4], pair=(2, 3), idx=5 -> [1, 5, 1, 2, 4]
    """
    i, n = 0, len(ids)
    res = []
    while i < n:
        if i < n-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            res.append(idx)
            i += 2
        else:
            res.append(ids[i])
            i += 1
    return res

def replace_control_characters(s: str) -> str:
    # given a string, we want to replace the control characters so that the 
    # print is nice looking
    chs = []
    for c in s:
        print(f"character: {c}:{unicodedata.category(c)[0]}")
        if unicodedata.category(c)[0] == 'C':
            # character is control character
            chs.append(f"\\u{ord(c):04x}")
        else:
            chs.append(c)
    return "".join(chs)

def render_tokens(t: bytes) -> str:
    decoded_str = t.decode('utf-8', errors='replace')
    return replace_control_characters(decoded_str)
    
    



