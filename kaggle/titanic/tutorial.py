#!/bin/env python
#_*_ encoding=utf-8 _*_

import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# machine learning
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

def drop_columns(df, columns):
	''' 删除无用字段
	'''
	return df.drop(columns, axis = 1)

def analysis_embarked(titanic_df):
	''' 分析Embarked Featrue

	发现三个港口生存和死亡比例相当, 可以推断对生存率没有啥影响
	'''
	# 港口分布
	embarked_counts = titanic_df['Embarked'].value_counts()

	# Survived
	survived = titanic_df[titanic_df.Survived == 1]
	survived_counts = survived['Embarked'].value_counts()

	# Dead分布
	dead = titanic_df[titanic_df.Survived == 0]
	dead_counts = dead['Embarked'].value_counts()

	# 展示
	fig,axes = plt.subplots(nrows=1, ncols=3)

	embarked_counts.plot(kind='bar', ax=axes[0]); axes[0].set_title('embarked')
	survived_counts.plot(kind='bar', ax=axes[1]); axes[1].set_title('survived')
	dead_counts.plot(kind='bar', ax=axes[2]); axes[2].set_title('dead')

	plt.show()	

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
	max = titanic_df.max()[feature]
	step = max / num if max > num else 1
	bins = range(0, max + step, step)

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
	plt.plot(bins, [0.5] * len(bins), color='r')    # 0.5 lines

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
	label_analysis(titanic_df, 'Cabin')

if __name__ == '__main__':
	run()
