#!/bin/env python
#_*_ encoding=utf-8 _*_
#refer: http://scikit-learn.org/stable/tutorial/basic/tutorial.html

from sklearn import datasets
from sklearn import svm

'''
Loading Dataset
'''
iris = datasets.load_iris()
digits = datasets.load_digits()

# digits.data gives access to the features that can be used to classify the digits samples
print digits.data     
print type(digits.data)    # numpy.ndarray
print digits.data.shape    # 1797 * 64

# digits.target gives the ground truth for the digit dataset, that is the number corresponding to each digit image that we are trying to learn
print digits.target    # [0 1 2 ..., 8 9 8] From 0 to 9
print type(digits.target)    # numpy.ndarray
print digits.target.shape     # 1797 * 1

'''
Learning and Predicting
'''
# SVM so easy
clf = svm.SVC(gamma=0.001, C=100.)

# produces a new array that contains all but the last entry of digits.data
clf.fit(digits.data[:-1], digits.target[:-1])

# Input = [[data1], [data2]]
print clf.predict(digits.data[-1:])    # 8
print clf.predict(digits.data[0:3])    # 0 1 2

'''
Model Persistence
'''
import pickle
from sklearn.externals import joblib

# pickle存入变量
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print clf2.predict(digits.data[0:3])

# joblib存入文件
joblib.dump(clf, 'svm.pkl')
clf3 = joblib.load('svm.pkl')
print clf3.predict(digits.data[0:3])

'''
Conventions
'''
import numpy as np
from sklearn import random_projection

# Type casting
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print X
print X.shape
print X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print X_new.dtype







