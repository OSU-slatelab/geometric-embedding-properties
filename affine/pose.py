import sys
import os

import numpy as np

dim = 10


u     = np.array([1 if i == 0 else 0 for i in range(dim)])
v     = np.array([1 if i == 1 else 0 for i in range(dim)])
u = np.array([u])
v = np.array([v])

print("vector")
print(u)
print("transpose")
print(np.transpose(u))
