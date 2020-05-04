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
iris_dataset = sklearn.datasets.load_iris()
