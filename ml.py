# import numpy as np

# array = np.array([13, 4, 4])
# print(array)

# tensor is a multidimensional array
# a gpu version of numpy ndarray

import numpy as np
import torch

# 2d array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# print(arr2d)

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# print(arr3d.ndim)

x = torch.rand(5, 3)
# print(x)

z = torch.zeros(5, 3)
# print(z)
# print(z.dtype)

torch.manual_seed(1729)
r1 = torch.rand(5, 2)
print(r1)
r2 = torch.rand(5, 2)
print(r2)
torch.manual_seed(1729)
r3 = torch.rand(5, 2)
print(r3)

data = [[1, 2], [3, 4]]
data_tensor = torch.tensor(data)
print(data_tensor[0])
