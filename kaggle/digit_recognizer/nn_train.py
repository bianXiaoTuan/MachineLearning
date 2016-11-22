#!/bin/env python
#_*_ encoding=utf-8 _*_

import numpy
import common
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

def run():
    # 载入训练数据
    x_train,y_train = common.load_train_data('./data/train.csv')

    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)

    # 载入测试数据
    x_test = common.load_test_data('./data/test.csv')
    x_test = numpy.array(x_test)

    # NN 训练
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(784, 2), random_state=1)
    clf.fit(x_train[0:10], y_train[0:10])   

    # 预测
    predict = clf.predict(x_test[0:10])
    print predict

    # 保存结果
    common.save_predict('./data/predict_nn.csv', predict)

if __name__ == '__main__':
    run()