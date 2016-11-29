#!/bin/env python
#_*_ encoding=utf-8 _*_

import numpy
import common
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

def run():
    # 载入训练数据
    X_train,y_train = common.load_train_data('./data/train.csv')
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)

    X_train,X_test,y_train,y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(X_train, y_train)

    print logistic.score(X_test, y_test)

if __name__ == '__main__':
    run()