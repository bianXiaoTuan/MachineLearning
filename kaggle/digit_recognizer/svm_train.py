#!/bin/env python
#_*_ encoding=utf-8 _*_
# SVM训练

import common

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA

def run():
    # 载入训练数据
    x_train,y_train = common.load_train_data('./data/train.csv')
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 每一幅图旋转90, 180, 270，扩充训练集
    x_train,y_train = common.rotate(x_train, y_train)

    # 降维度
    COMPONENT_NUM = 50
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(x_train)
    x_train = pca.transform(x_train)

    # SVM 训练
    svc = SVC()
    svc.fit(x_train, y_train)

    # 载入测试数据
    x_test = common.load_test_data('./data/test.csv')
    x_test = np.array(x_test)

    x_test = pca.transform(x_test)

    # 预测
    predict = svc.predict(x_test)

    # 保存结果
    common.save_predict('./data/predict_svm.csv', predict)

if __name__ == '__main__':
    run()