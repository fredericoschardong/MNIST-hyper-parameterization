import os
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

#supress warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

#load data
dataset = loadmat('exdata.mat')

#prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(dataset['X'].T, dataset['y'].T, stratify=dataset['y'].T, random_state = 1, test_size = 0.2)

def test_normalizers():
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer
    from sklearn.preprocessing import QuantileTransformer

    global X_train, X_test, y_train, y_test

    #histogram
    for i in range(X_train.shape[0]):
        x = X_train[i]
        hist, bins = np.histogram(x)
        plt.plot(bins[:hist.size], hist / np.sum(hist))
        print(i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

    plt.xlabel('Values')
    plt.ylabel('Proportions')
    plt.savefig('Histogram before normalization.png')
    plt.clf()

    print('X_train: min %.2f max %.2f mean %.2f std %.2f' % (np.min(X_train), np.max(X_train), np.mean(X_train), np.std(X_train)))

    for scaler in [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), QuantileTransformer(output_distribution='normal'), QuantileTransformer(output_distribution='uniform'), Normalizer()]:
        #normalization
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #new histogram with normalized data
        for i in range(X_train.shape[0]):
            x = X_train[i]
            hist, bins = np.histogram(x)
            plt.plot(bins[:hist.size], hist / np.sum(hist))
            print(scaler, i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

        plt.xlabel('Values')
        plt.ylabel('Proportions')
        plt.savefig('Histogram after normalization with %s.png' % scaler)
        plt.clf()

        start = time.time()

        #parameter search space
        parameters = {'hidden_layer_sizes': [(100,)],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'alpha': [0.00001, 0.0001, 0.001, 0.01],
                      'random_state': [1],
                      'max_iter': [100],
                      'solver': ['lbfgs', 'sgd', 'adam']}

        #use f1 to rank parameters, all cores and 5-cross fold validation
        clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
        clf.fit(X_train, y_train.ravel())
        y_true, y_pred = y_test, clf.predict(X_test)

        print(classification_report(y_true, y_pred))
        print(clf.best_params_)
        print("total time for scaler", scaler, time.time() - start)

def test_1():
    global X_train, X_test, y_train, y_test

    for scaler in [MinMaxScaler(), MaxAbsScaler()]:
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for s in ['lbfgs', 'sgd', 'adam']:
            for h in [(10,10,10,10,10,10,10,10,10,10), (20,20,20,20,20), (25,25,25,25)]:
                start = time.time()

                parameters = {'hidden_layer_sizes': [h],
                              'activation': ['logistic', 'tanh'],
                              'alpha': [0.00001, 0.0001, 0.001, 0.01],
                              'random_state': [1],
                              'max_iter': [100],
                              'solver': [s]}

                clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
                clf.fit(X_train, y_train.ravel())
                y_true, y_pred = y_test, clf.predict(X_test)

                print(classification_report(y_true, y_pred))
                print(clf.best_params_)
                print("total time for scaler", scaler, time.time() - start)

def test_2():
    global X_train, X_test, y_train, y_test

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    for s in ['lbfgs', 'sgd', 'adam']:
        for h in [(25,), (25,25), (25,25,25), (25,25,25,25), (25,25,25,25,25), (25,25,25,25,25,25), (25,25,25,25,25,25,25)]:
            start = time.time()

            parameters = {'hidden_layer_sizes': [h],
                          'activation': ['tanh'],
                          'alpha': [0.01],
                          'random_state': [1],
                          'max_iter': [1000],
                          'solver': [s]}

            clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
            clf.fit(X_train, y_train.ravel())
            y_true, y_pred = y_test, clf.predict(X_test)

            print(classification_report(y_true, y_pred))
            print(clf.best_params_)
            print(confusion_matrix(y_true, y_pred))
            print("total time", time.time() - start)

def test_best_deep_result_confusion_matrix():.
    from sklearn.metrics import confusion_matrix
    
    global X_train, X_test, y_train, y_test

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    parameters = {'hidden_layer_sizes': [(25,25,25)],
                  'activation': ['tanh'],
                  'alpha': [0.01],
                  'random_state': [1],
                  'max_iter': [1000],
                  'solver': ['adam']}

    clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
    clf.fit(X_train, y_train.ravel())
    y_true, y_pred = y_test, clf.predict(X_test)

    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(labels)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig('Confusion matrix.png')
    plt.clf()

test_normalizers()
test_1()
test_2()
test_best_deep_result_confusion_matrix()
