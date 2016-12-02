#!/bin/env python
#_*_ encoding=utf-8 _*_

from common import *

def run():
	''' Logistic Regression
	'''
	titanic_df,test_df = load_data()

	# Feature加工
	X_train,y_train,X_test = feature_minning(titanic_df, test_df)

	X_train,X_cv,y_train,y_cv = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

	logistic = LogisticRegression(C=1e5)
	logistic.fit(X_train, y_train) 

	# 交叉验证打分 
	print logistic.score(X_train, y_train)
	print logistic.score(X_cv, y_cv)

	# Predict
	predict = logistic.predict(X_test)
	print predict

	save_predict('./data/predict_lr.csv', predict)

if __name__ == '__main__':
	run()
