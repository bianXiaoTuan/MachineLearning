#!/bin/env python
#_*_ encoding=utf-8 _*_
# SVM训练

from common import *
import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score

def run():
    # Load Data
    titanic_df,test_df = load_data()

    # Feature加工
    X_train,y_train,X_test = feature_minning(titanic_df, test_df)

    # 拆分数据
    X_train,X_cv,y_train,y_cv = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    # SVM 训练
    svc = SVC()

    # Grid Search Train
    print 'Grid Search'

    param_grid = [ {
        # 'kernel': ['rbf', 'linear', 'sigmoid', 'poly']
        # 'kernel': ['rbf'],
        # 'C': np.logspace(-3, 2, 10),
        # 'gamma': np.logspace(-3, 2, 10)

        'kernel': ['poly'],
        'degree': [1, 2, 3, 4, 5],
        'coef0': np.logspace(-3, 2, 10)
    }]

    clf = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Scores
    print 'Scores'
    print clf.best_score_
    print clf.best_params_
    print clf.score(X_cv, y_cv)

if __name__ == '__main__':
    run()
