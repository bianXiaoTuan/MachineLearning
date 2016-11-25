#!/bin/env python
#_*_ encoding=utf-8 _*_
# SVM训练

import common

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def run():
    # 载入训练数据
    print 'Load Data'
    x_train,y_train = common.load_train_data('./data/train.csv')
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 每一幅图旋转90, 180, 270，扩充训练集
    x_train,y_train = common.rotate(x_train, y_train)

    # 拆分数据
    print 'Split Data'
    x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    # 降维度
    print 'PCA'
    COMPONENT_NUM = 50
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # SVM 训练
    print 'SVM Training'
    # svc = SVC(kernel='sigmoid', C=1.0, gamma='auto')
    svc = SVC()
    svc.fit(x_train, y_train)

    # 预测打分
    print 'SVM Predict'

    predict = svc.score(x_test, y_test)
    print 'Test %f\n' % predict

    predict = svc.score(x_train, y_train)
    print 'Train %f\n' % predict

    # # 载入测试数据
    # print 'Load Test Data'
    # x_test = common.load_test_data('./data/test.csv')
    # x_test = np.array(x_test)
    # x_test = pca.transform(x_test)

    # # 保存结果
    # common.save_predict('./data/predict_svm.csv', predict)

if __name__ == '__main__':
    run()
