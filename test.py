"""
import torch
from collections import namedtuple, deque
import numpy as np
print(torch.cuda.is_available())
reward = 5


memory = deque(maxlen=10)

for i in range(10) :
    memory.append(i)
print(memory)

a = np.array([1,3])
print([memory[i] for i in a])

import torch

print(torch.cuda.is_available())
"""

import numpy as np

