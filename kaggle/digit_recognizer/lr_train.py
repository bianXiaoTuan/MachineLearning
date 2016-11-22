#!/bin/env python
#_*_ encoding=utf-8 _*_

import numpy
import common
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier

def run():
    # 载入训练数据
    x_train,y_train = common.load_train_data('./data/train.csv')

    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)

    print x_train.shape
    print y_train.shape

    # 载入测试数据
    x_test = common.load_test_data('./data/test.csv')
    x_test = numpy.array(x_test)

    print x_test.shape

    # LR 训练
    lr_classif = OneVsRestClassifier(estimator=linear_model.LogisticRegression(C=1e5))
    lr_classif.fit(x_train, y_train)

    # 预测
    predict = lr_classif.predict(x_test)

    # 保存结果
    common.save_predict('./data/predict_lr.csv', predict)

if __name__ == '__main__':
    run()