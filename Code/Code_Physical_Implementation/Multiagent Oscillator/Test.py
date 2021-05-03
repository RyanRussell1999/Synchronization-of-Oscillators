import numpy as np
x = np.array([0, 1])
data = np.array([x[1], 2, 7])
data = np.array2string(data, precision=2, separator=',',
                      suppress_small=True)
print(data)

data = np.fromstring(data[1:-1], sep=',')

print(data)