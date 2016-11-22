#!/bin/env python
#_*_ encoding=utf-8 _*_
#公共函数

import numpy as np

def load_train_data(file):
    ''' 加载文件
    '''
    y_train = []
    x_train = []

    with open(file, 'r') as reader:
        reader.readline()

        for line in reader.readlines():
            data = list(map(int, line.rstrip().split(',')))
            y_train.append(data[0])
            x_train.append(data[1:])

    return x_train,y_train 

def load_test_data(file):
    ''' 载入测试数据
    '''
    x_test = []

    with open(file, 'r') as reader:
        reader.readline()

        for line in reader.readlines():
            data = list(map(int, line.rstrip().split(',')))
            x_test.append(data)

    return x_test 

def save_predict(file, predict):
    ''' 记录结果
    '''
    with open(file, 'w') as writer:
        writer.write('"ImageId","Label"\n')
        count = 0
        for p in predict:
            count += 1
            writer.write(str(count) + ',' + str(p) + '\n')

def rotate(x, y):
    ''' 每个图转90, 180, 270, 扩容训练集
    '''
    new_x = []
    new_y = []
    length = len(x)

    for i in range(length):
        data = x[i].reshape((28, 28))
        label = y[i]

        new_x.append(data.ravel())
        new_x.append(rotate90(data).ravel())
        new_x.append(rotate180(data).ravel())
        new_x.append(rotate270(data).ravel())

        new_y.append(label)
        new_y.append(label)
        new_y.append(label)
        new_y.append(label)

    return np.array(new_x),np.array(new_y)

def rotate90(x):
    ''' 旋转90度
    '''
    return np.rot90(x)

def rotate180(x):
    ''' 旋转180度
    '''
    return np.rot90(np.rot90(x))

def rotate270(x):
    ''' 旋转270度
    '''
    return np.rot90(np.rot90(np.rot90(x)))
