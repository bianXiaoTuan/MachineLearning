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
	'''
	titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

	print type(titanic_df)

	print titanic_df[['Embarked', 'Survived']]
	print type(titanic_df[['Embarked', 'Survived']])

	# # Embarked数量分布
	# embarked_info = np.array(titanic_df['Embarked'])
	# types,counts = np.unique(embarked_info, return_counts=True)

	# plt.title('Embarked Distribute')

	# ypos = np.arange(len(types))
	# plt.bar(ypos, counts, align='center', alpha=0.5)
	# plt.xticks(ypos, types)
	# plt.ylabel('counts')
	# plt.show()

	# # Embarked Suivived分布
	# embarked_info = [row['Embarked'] for row in titanic_df if row['Survived'] == 1]
	# embarked_info = np.array(embarked_info)
	# types,counts = np.unique(embarked_info, return_counts=True)

	# plt.title('Survived Embarked Distribute')

	# ypos = np.arange(len(types))
	# plt.bar(ypos, counts, align='center', alpha=0.5)
	# plt.xticks(ypos, types)
	# plt.ylabel('Survived Counts')
	# plt.show()

def run():
	''' 分析流程
	'''
	# 加载数据
	titanic_df,test_df = load_data()

	# 丢弃无用字段
	titanic_df = drop_columns(titanic_df, ['PassengerId', 'Name', 'Ticket'])
	test_df = drop_columns(test_df, ['Name', 'Ticket'])

	# 分析Embarked
	analysis_embarked(titanic_df)

if __name__ == '__main__':
	run()


