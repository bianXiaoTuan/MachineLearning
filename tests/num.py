#!/bin/env python
#_*_ encoding=utf-8 _*_
#refer: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

import numpy as np

'''
Basics
'''
a = np.arange(15).reshape(3, 5)
print a.ndim    # number of axes 2
print a.shape    # dimensions (3, 5)
print a.size    # number of elements 15
print a.dtype    # type of element int64
print a.itemsize    # size in bytes of each element

'''
Array Creation
'''
a = np.array([2, 3, 4])    # 一维数组
print a

a = np.array([(1, 2, 3), (4, 5, 6)])    # 多维数组
print a

a = np.zeros((3, 4))    # 元素全部为0 矩阵
print a 

a = np.ones((2, 3, 4), dtype=np.int16)    # 元素全部为1 矩阵
print a 

a = np.random.random((3, 3))    # [0 1]范围内随机矩阵
print a

'''
Basic Operations
'''
a = np.array( [20,30,40,50] )
b = np.arange( 4 )     # [1, 2, 3, 4]
c = a - b
print c
print b ** 2    # 对每个元素操作
print a < 35    # bool array

A = np.array([(1, 1), (0, 1)])
B = np.array([(2, 0), (3, 4)])
print A * B    # 对应元素相乘
print np.dot(A, B)    # 矩阵乘法

a = np.random.random((2, 3))
print a
print a.sum()    # 求和
print a.min()    # 最小值
print a.max()    # 最大值

a = np.arange(12).reshape(3, 4)
print a
print a.sum(axis=0)    # 每一列的和
print a.min(axis=1)    # 每一行的和
print a.cumsum(axis=1)    # cmmulative sum along each row

'''
Universal Functions
'''
a = np.arange(12).reshape(3, 4)
print np.exp(a)
print np.sqrt(a)

'''
Indexing, Slicing and Iterating
'''
a = np.arange(10)
print a[2]    # 2
print a[2:5]    # [2 3 4]

# From start to position 6, exclusive, set every 2nd element to -100
a[:6:2] = -100    # [-100  1  -100  3  -100  5  6  7  8  9]
print a

# 反转
a = np.arange(12)
print a[ : :-1] 

# 创建多维数组
def f(x, y):
    return 10 * x + y
a = np.fromfunction(f, (3, 3), dtype=int)
print a

# 切片
a = np.arange(12).reshape(3, 4)    # [[0 1 2 3], [4 5 6 7], [8 9 10 11]]
print a[0:2, :]    # [[0 1 2 3], [4 5 6 7]]
print a[:, 1]    # [1 5 9]
print a[-1]    # [8 9 10 11]

# 迭代
a = np.arange(12).reshape(3, 4)    # [[0 1 2 3], [4 5 6 7], [8 9 10 11]]

for row in a:    # 打印每一行
    print row

for elem in a.flat:    # 一个一个打印元素
    print elem

'''
Shape Manipulation
'''
a = np.floor(10 * np.random.random((3, 4)))
print a
print a.shape    # (3, 4)

print a.ravel()    # flatten the array








