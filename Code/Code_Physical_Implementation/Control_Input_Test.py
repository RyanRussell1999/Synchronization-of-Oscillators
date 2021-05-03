import numpy as np

u = 1
k = 1
while k < 100:
    if np.remainder(k,5) == 0:
        u = -u

    print(u)
    k = k + 1