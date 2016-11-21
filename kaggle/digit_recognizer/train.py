#!/bin/env python
#_*_ encoding=utf-8 _*_

import csv
import random
import numpy as np
from scipy import io

def load_train_data(file):
    '''
    '''
    f = open(file, 'rb')
    reader = csv.reader(f)

    # Remove first line
    data = [row for row in reader]
    data = [row for row in data[1:]]

    # Total Data
    x = np.mat([[int(elem) for elem in row[1:]] for row in data])
    y = np.mat([int(row[0]) if int(row[0]) != 0 else 10 for row in data])
    y = y.conj().transpose()

    # Shuffle
    random.shuffle(data)

    total_count = len(data)
    train_up_index = total_count / 100 * 70
    cv_up_index = total_count

    # Train Data
    x_train = np.mat([[int(elem) for elem in row[1:]] for row in data[0:train_up_index]])
    y_train = np.mat([int(row[0]) if int(row[0]) != 0 else 10 for row in data[0:train_up_index]])
    y_train = y_train.conj().transpose()

    # Cross validated Data
    x_cv = np.mat([[int(elem) for elem in row[1:]] for row in data[train_up_index:cv_up_index]])
    y_cv = np.mat([int(row[0]) if int(row[0]) != 0 else 10 for row in data[train_up_index:cv_up_index]])
    y_cv = y_cv.conj().transpose()

    print np.shape(x)
    print np.shape(y)
    io.savemat('./data/x.mat', {'data': x})
    io.savemat('./data/y.mat', {'data': y})

    print np.shape(x_train)
    print np.shape(y_train)
    io.savemat('./data/x_train.mat', {'data': x_train})
    io.savemat('./data/y_train.mat', {'data': y_train})

    print np.shape(x_cv)
    print np.shape(y_cv)
    io.savemat('./data/x_cv.mat', {'data': x_cv})
    io.savemat('./data/y_cv.mat', {'data': y_cv})

def run():
    ''' 运行
    '''
    load_train_data('./data/train.csv')

    drawLearningCurve()

    drawLambdaCurve()

    train_by_logistic_regress()

    predict_by_logistic_regress()

    build_result()

if __name__ == '__main__':
    load_data()
    #generate_result()
	# find_change()
