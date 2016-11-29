#!/bin/env python
#_*_ encoding=utf-8 _*_
#公共函数

import re
import numpy as np
import pandas as pd

def load_data(file):
    ''' 加载文件
    '''
    train_fd = pd.read_csv(file)

    # 删除无用columns
    train_fd = train_fd.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    print train_fd['Embarked'].fillna("S")

def remove_name(str):
    ''' 删除名字

    @param: str (sring) e.g. '1,1,1,"Behr, Mr. Karl Howell",male,26,0,0,111369,30,C148,C'
    @return ([]) e.g. ['1', '1', '1', 'male', '26', '0', '0', '111369', '30', 'C148', 'C']
    '''
    str_r = str.split('"')
    str = str_r[0].strip(',') + ',' + str_r[2].strip(',')

    return str.split(',')

def load_train_data(file):
    ''' 加载文件
    '''
    y_train = []
    x_train = []

    with open(file, 'r') as reader:
        reader.readline()

        for line in reader.readlines()[0:10]:
            data = remove_name(line.strip())

            # x_train
            p_class = int(data[2])
            sex = 1 if data[3] == 'male' else 0
            age = 0 if data[4] == '' else int(data[4])
            sib_sp = int(data[5])
            parch = int(data[6])
            fare = float(data[8])

            x_train.append([p_class, sex, age, sib_sp, parch, fare])

            # y_train
            survived = data[1]
            y_train.append(int(survived))

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

if __name__ == '__main__':
    # remove_name('890,1,1,"Behr, Mr. Karl Howell",male,26,0,0,111369,30,C148,C')

    # x_train,y_train = load_train_data('data/train.csv')
    # print x_train
    # print y_train
    load_data('data/train.csv')
