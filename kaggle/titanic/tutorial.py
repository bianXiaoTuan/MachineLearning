#!/bin/env python
#_*_ encoding=utf-8 _*_

import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def load_data():
	''' 加载数据
	'''
	titanic_df = pd.read_csv('./data/train.csv')
	test_df    = pd.read_csv('./data/test.csv')
	return titanic_df,test_df	

def label_analysis(titanic_df, feature):
	''' 对feature中label量进行分析
	'''
	titanic_df[feature] = titanic_df[feature].fillna('')

	# Survived
	survived = titanic_df[titanic_df.Survived == 1]
	survived_counts = survived[feature].value_counts().sort_index()
	survived_counts.index = survived_counts.index.astype(str)

	# Dead分布
	dead = titanic_df[titanic_df.Survived == 0]
	dead_counts = dead[feature].value_counts().sort_index()
	dead_counts.index = dead_counts.index.astype(str)

	print survived_counts
	print dead_counts

	# Survived Counts / Total, index不一致
	result = [survived_counts[i] / float(survived_counts[i] + dead_counts[i]) for i in range(len(survived_counts))]
	print result
	print survived_counts.index

	result = pd.DataFrame(result, index=survived_counts.index).fillna(0)
	print result

	result.plot(kind='bar').set_title('%s Survived Rate' % feature)
	plt.show()

def numeric_analysis(titanic_df, feature, num=10):
	''' 对feature中数值量进行分析
	'''
	titanic_df[feature] = titanic_df[feature].fillna(0)
	titanic_df[feature] = titanic_df[feature].astype(int)

	# Bins
	min = titanic_df.min()[feature]
	max = titanic_df.max()[feature]
	step = (max - min) / num if (max - min) > num else 1
	bins = range(min - step, max + step, step)

	# Survived Counts
	survived_cats = pd.cut(titanic_df[feature][titanic_df.Survived == 1], bins)
	survived_counts = pd.value_counts(survived_cats).sort_index()
	survived_counts.index = survived_counts.index.astype(str)

	# Dead Counts
	dead_cats = pd.cut(titanic_df[feature][titanic_df.Survived == 0], bins)
	dead_counts = pd.value_counts(dead_cats).sort_index()
	dead_counts.index = dead_counts.index.astype(str)

	# Survived Counts / Total
	result = [survived_counts[i] / float(survived_counts[i] + dead_counts[i]) for i in range(len(survived_counts))]
	result = pd.DataFrame(result, index=bins[1:]).fillna(0)

	result.plot(kind='bar').set_title('%s Survived Rate' % feature)

	# 50% standard line
	plt.plot(bins, [0.5] * len(bins), color='r')

	plt.show()

def run():
	''' 分析流程
	'''
	# 加载数据
	titanic_df,test_df = load_data()

	# numeric_analysis(titanic_df, 'Age', num=20)
	# numeric_analysis(titanic_df, 'Fare', num=20)
	# numeric_analysis(titanic_df, 'SibSp', num=5)
	# numeric_analysis(titanic_df, 'Parch', num=5)

	# label_analysis(titanic_df, 'Embarked')
	# label_analysis(titanic_df, 'Pclass')
	# label_analysis(titanic_df, 'Sex')
	# label_analysis(titanic_df, 'Cabin')

	titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'],  axis=1)

	# Keep Age
	mean_age = titanic_df.mean()['Age']
	titanic_df['Age'] = titanic_df['Age'].fillna(mean_age).astype(int)

	# Fare
	titanic_df['Fare'] = titanic_df['Fare'].astype(int)

	# Sex
	titanic_df.loc[titanic_df['Sex'] == 'female', 'Sex'] = 0
	titanic_df.loc[titanic_df['Sex'] == 'male', 'Sex'] = 1

	# SibSp and Parch -> Family
	titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"]
	titanic_df.loc[titanic_df['Family'] > 0, 'Family'] = 1
	titanic_df.loc[titanic_df['Family'] == 0, 'Family'] = 0

	# Drop SibSp and Parch
	titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

	X = titanic_df.drop("Survived",axis=1)
	y = titanic_df["Survived"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	logistic = LogisticRegression(C=1e5)
	logistic.fit(X_train, y_train) 

	# 打分 
	print logistic.score(X_train, y_train)
	print logistic.score(X_test, y_test)

if __name__ == '__main__':
	run()
