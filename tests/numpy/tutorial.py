#!/bin/env python
#_*_ encoding=utf-8 _*_
#refer: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

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

print a.ravel()    # flatten the array [ 1.  2.  0.  6.  4.  5.  8.  8.  0.  1.  2.  5.]

a = np.arange(12).reshape(3, 4)
a.shape = (2, 6)    # 将a 转化成 2 * 6的多维矩阵, 改变a 本身
print a

a.resize((6, 2))    # 将a 转化成 6 * 2的多维矩阵, 改变a 本身
print a

print a.reshape(3, -1)    # 不改变a
print a

'''
Stacking Together Different Arrays
'''
a = np.floor(10 * np.random.random((2, 2)))
print a  

b = np.floor(10 * np.random.random((2, 2)))
print b  

print np.vstack((a, b))    # 按照行拼接
print np.hstack((a, b))    # 按照列拼接

print np.column_stack((a,b))   # With 2D arrays

a = np.array([4.,2.])
print a[:,newaxis]  # This allows to have a 2D columns vector

b = np.array([2.,8.])
print np.column_stack((a[:,newaxis],b[:,newaxis]))
print np.vstack((a[:,newaxis],b[:,newaxis])) 

'''
Splitting one  array into several smaller ones
'''
a = np.floor(10 * np.random.random((2,12)))
print a

print np.hsplit(a, 3)
print np.hsplit(a, (3, 4))

'''
Copies and Views
'''
a = np.arange(12)
b = a
print b is a    # True

c = a.view()    # 浅拷贝
print c is a    # False
c[1] = 100
print a

d = a.copy()    # 深拷贝
print d is a    # False

'''
旋转图像
'''
x_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 255, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 250, 253, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 248, 253, 167, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 247, 253, 208, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 207, 253, 235, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 209, 253, 253, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 254, 253, 238, 170, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 210, 254, 253, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 209, 253, 254, 240, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 253, 253, 254, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 206, 254, 254, 198, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168, 253, 253, 196, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 203, 253, 248, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 188, 253, 245, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 103, 253, 253, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 240, 253, 195, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 220, 253, 253, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 253, 253, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 251, 253, 250, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 214, 218, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
x_train = np.array(x_train).reshape((28, 28))

plt.imshow(np.rot90(np.rot90(np.rot90(x_train))))
plt.show()