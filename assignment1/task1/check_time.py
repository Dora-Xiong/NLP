import numpy
from collections import Counter, defaultdict
import random
import time
import matplotlib.pyplot as plt
import string

def flatten_list(nested_list: list):
    if len(nested_list) == 0:
        return []
    return numpy.concatenate(nested_list).tolist()

def flatten_list_v1(nested_list: list):
    if len(nested_list) == 0:
        return []
    return [item for sublist in nested_list for item in sublist]

def flatten_list_v2(nested_list: list):
    return numpy.hstack(nested_list).tolist()
    

def list_generater(length: int):
    return [[''.join(random.choices(string.ascii_lowercase, k=random.randint(1, 4))) for _ in range(random.randint(2, 6))] for _ in range(length)]

def char_count(s: str):
    return dict(Counter(s))

def char_count_v1(s: str):
    count = defaultdict(int)
    for char in s:
        count[char] += 1
    return dict(count)

def char_count_v2(s: str):
    return {char: s.count(char) for char in set(s)}

    

def string_generater(length: int):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Test the running time of flatten_list
x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    nested_list = list_generater(10**i)
    begin = time.time()
    flatten_list(nested_list)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
plt.plot(x_s, y_s)
plt.xlabel('Length of the list')
plt.ylabel('Running time')
plt.title('Running time of flatten_list')
plt.show()

x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    nested_list = list_generater(10**i)
    begin = time.time()
    flatten_list_v1(nested_list)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.plot(x_s, y_s)
plt.xlabel('Length of the list')
plt.ylabel('Running time')
plt.title('Running time of flatten_list_v1')
plt.show()

x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    nested_list = list_generater(10**i)
    begin = time.time()
    flatten_list_v2(nested_list)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.plot(x_s, y_s)
plt.xlabel('Length of the list')
plt.ylabel('Running time')
plt.title('Running time of flatten_list_v2')
plt.show()

# Test the running time of char_count
x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    s = string_generater(10**i)
    begin = time.time()
    char_count(s)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.plot(x_s, y_s)
plt.xlabel('Length of the string')
plt.ylabel('Running time')
plt.title('Running time of char_count')
plt.show()

x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    s = string_generater(10**i)
    begin = time.time()
    char_count_v1(s)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.plot(x_s, y_s)
plt.xlabel('Length of the string')
plt.ylabel('Running time')
plt.title('Running time of char_count_v1')
plt.show()

x_s = [10**i for i in range(8)]
y_s = []
for i in range(8):
    s = string_generater(10**i)
    begin = time.time()
    char_count_v2(s)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.plot(x_s, y_s)
plt.xlabel('Length of the string')
plt.ylabel('Running time')
plt.title('Running time of char_count_v2')
plt.show()


# Compare the running time of different implementations of flatten_list
x_s = ["A", "B", "C"]
y_s = []
nested_list = list_generater(10**6)
for i in range(3):
    begin = time.time()
    if i == 0:
        flatten_list(nested_list)
    elif i == 1:
        flatten_list_v1(nested_list)
    else:
        flatten_list_v2(nested_list)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))

plt.bar(x_s, y_s)
plt.xlabel('Implementation')
plt.ylabel('Running time')
plt.title('Running time of different versions of flatten_list')
plt.show()

# Compare the running time of different implementations of char_count
x_s = ["A", "B", "C"]
y_s = []
s = string_generater(10**6)
for i in range(3):
    begin = time.time()
    if i == 0:
        char_count(s)
    elif i == 1:
        char_count_v1(s)
    else:
        char_count_v2(s)
    end = time.time()
    y_s.append(end - begin)
    print("The runnining time = {}.".format(end - begin))
    
plt.bar(x_s, y_s)
plt.xlabel('Implementation')
plt.ylabel('Running time')
plt.title('Running time of different versions of char_count')
plt.show()