# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:30:35 2017

@author: minven2
"""

from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn import linear_model  
import time

x,y = make_multilabel_classification(sparse = True, n_samples=10000, n_features=10, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)
                                         
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=SVC(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 201.42 seconds


x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=10, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=SVC(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 3hours not enough



x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=10, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)

start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=linear_model.SGDClassifier(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 14.51s


x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=100, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=linear_model.SGDClassifier(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 31.26s

x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=1000, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=linear_model.SGDClassifier(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 35.26s

x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=10000, n_classes=100, n_labels=3,
                                          return_indicator = 'sparse', allow_unlabeled = False)
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=linear_model.SGDClassifier(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 46.91s

x,y = make_multilabel_classification(sparse = True, n_samples=100000, n_features=10000, n_classes=1000, n_labels=50,
                                          return_indicator = 'sparse', allow_unlabeled = False)
start_time_classification = time.time()                                 
classifier = ClassifierChain(classifier=linear_model.SGDClassifier(),require_dense = [False, True])
classifier.fit(x, y)
print("Tokenization %.2f seconds" % (time.time() - start_time_classification))
# 607.42 seconds