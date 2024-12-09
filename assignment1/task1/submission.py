import numpy
from collections import Counter, defaultdict

def flatten_list(nested_list: list):
    # 方法1
    if len(nested_list) == 0:
        return []
    return numpy.concatenate(nested_list).tolist()

    # 方法2
    # if len(nested_list) == 0:
    #     return []
    # return [item for sublist in nested_list for item in sublist]
    
    # 方法3
    # return numpy.hstack(nested_list).tolist()




def char_count(s: str):
    # 方法1
    return dict(Counter(s))
    
    # 方法2
    # count = defaultdict(int)
    # for char in s:
    #     count[char] += 1
    # return dict(count)
    
    # 方法3
    # return {char: s.count(char) for char in set(s)}