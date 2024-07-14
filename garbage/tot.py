from collections import namedtuple, deque

t = deque([], maxlen=3)
for i in range(10):
    t.append(i)
    print(t)