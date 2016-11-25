#!/bin/env python
#_*_ encoding=utf-8 _*_

import common

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def run():
    # 载入训练数据
    print 'Load Data'
    x_train,y_train = common.load_train_data('./data/train.csv')
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 拆分数据
    print 'Split Data'
    x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, test_size=0.01, random_state=0)

    # NUM = 1000
    # x_train = x_train[0:NUM, :]
    # y_train = y_train[0:NUM]

    # x_test = x_train[-20:, :]
    # y_test = y_train[-20:]

    # 降维度
    print 'PCA'
    COMPONENT_NUM = 50
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # SVM
    print 'SVM'
    # svc = svm.SVC(C=1.0, kernel='linear')
    svc = svm.SVC()

    # Grid Search Train
    print 'Grid Search'

    # Model
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # clf = GridSearchCV(estimator=svc, param_grid=dict(kernel=kernels), n_jobs=-1)

    # C
    Cs = np.logspace(-6, 2, 50)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)
    clf.fit(x_train, y_train)

    print 'Scores'
    print clf.best_score_
    print clf.best_estimator_.kernel    # 效果最好C
    print clf.best_estimator_.C    # 效果最好C

    print clf.score(x_test, y_test)

if __name__ == '__main__':
    run()
