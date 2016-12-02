#!/bin/env python
#_*_ encoding=utf-8 _*_
# SVM训练

from common import *
import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def run():
    # Load Data
    titanic_df,test_df = load_data()

    # Feature加工
    X_train,y_train,X_test = feature_minning(titanic_df, test_df)

    # 拆分数据
    X_train,X_cv,y_train,y_cv = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    # SVM 训练
    print 'SVM Training'
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)

    # 预测打分
    print 'SVM Predict'
    predict = svc.score(X_train, y_train)
    print 'Train %f\n' % predict

    predict = svc.score(X_cv, y_cv)
    print 'CV %f\n' % predict

    # Predict
    predict = svc.predict(X_test)
    print predict

    # Save
    save_predict('./data/predict_svm.csv', predict) 

if __name__ == '__main__':
    run()
