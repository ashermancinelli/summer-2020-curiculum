import numpy as np
import matplotlib.pyplot as plt

'''
Scikit Learn is a very common libary for classification
and simpler machine learning tasks.

Iris is a dataset which contains information about several
kinds of flowers. We will create a few models which will
infer the kind of flower from the flower's qualities.
'''
import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
iris_dataset = datasets.load_iris()
x = iris_dataset['data']
y = iris_dataset['target']

# Randomize data
x, y = shuffle(x, y)

def print_results(predictions, actual):
    print('Prediction\tActual')
    print('-' * 30)
    for i, j in zip(predictions, test_y):
        print(iris_dataset['target_names'][i],
              '\t',
              iris_dataset['target_names'][j])

    failures = sum(predictions - test_y != 0)
    print(f'\nTotal failures: {failures}')
    print(f'Accuracy: {failures / len(predictions)}')

if __name__ == '__main__':
    # Grab 80% of data for training
    train_x = x[:int(len(x)*.8)]
    train_y = y[:int(len(y)*.8)]

    # Use the rest for testing
    test_x = x[int(len(x)*.8):]
    test_y = y[int(len(y)*.8):]

    from sklearn import svm
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    print_results(predictions, test_y)
