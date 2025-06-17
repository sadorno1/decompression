import numpy as np

array1 = np.array([[1], [2], [3]])
array2 = np.array([[4], [5], [6]])
print(f'2 {array2.shape}')

array3 = np.array([[7], [8], [9]])
stack = np.hstack((array1, array2, array3))

print(stack)