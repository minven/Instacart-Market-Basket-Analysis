# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:09:35 2017

@author: minven2
"""

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
if False:
    # this will generate a
    x, y = make_multilabel_classification(sparse = True, n_labels = 5,
  return_indicator = 'sparse', allow_unlabeled = False)
    
if False:

    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    
    # learn the classifier
    classifier = MLkNN(k=3)
    classifier.fit(X_train, y_train)
    
    # predict labels for test data
    predictions = classifier.predict(X_test)
    predictions.toarray()
    
    
    
if False:
    x, y = make_multilabel_classification(sparse = True, n_labels = 5,
                                          return_indicator = 'sparse', allow_unlabeled = False)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    # Classifier Chains Multi-Label Classifier 
    classifier = ClassifierChain(classifier=SVC())
    classifier.fit(X_train, y_train)
    # predict labels for test data
    predictions = classifier.predict(X_test)
    predictions.toarray()
    
    
    
    
#    X_extended = self.ensure_input_format(
#        X, sparse_format='csc', enforce_sparse=True)
#    y = self.ensure_output_format(
#        y, sparse_format='csc', enforce_sparse=True)
    import copy
    from numpy import ravel
    label_count = y.shape[1]
    classifiers = [None for X_train in range(label_count)]

    for label in range(label_count):
        classifier = copy.deepcopy(SVC())
        y_subset = generate_data_subset(y, label, axis=1)

        classifiers[label] = classifier.fit(X_train,y_subset)
        #X_extended = hstack([X_extended, y_subset])
        
        
def generate_data_subset(y, subset, axis):
    return_data = None
    if axis == 1:
        return_data = y.tocsc()[:, subset]
    elif axis == 0:
        return_data = y.tocsr()[subset, :]

    return return_data




from scipy.sparse import issparse, csr_matrix
def ensure_input_format(X, sparse_format='csr', enforce_sparse=False):
    """Ensure the desired input format

    This function ensures that input format follows the density/sparsity requirements of base classifier. 

    Parameters
    ----------

    X : array-like or sparse matrix, shape = [n_samples, n_features]
        An input feature matrix

    sparse_format: string
        Requested format of returned scipy.sparse matrix, if sparse is returned

    enforce_sparse : bool
        Ignore require_dense and enforce sparsity, useful internally

    Returns
    -------

    transformed X : array-like or sparse matrix, shape = [n_samples, n_features]
        If require_dense was set to true for input features in the constructor, 
        the returned value is an array-like of array-likes. If require_dense is 
        set to false, a sparse matrix of format sparse_format is returned, if 
        possible - without cloning.
    """
    is_sparse = issparse(X)

    if is_sparse:
        if self.require_dense[0] and not enforce_sparse:
            return X.toarray()
        else:
            if sparse_format is None:
                return X
            else:
                return get_matrix_in_format(X, sparse_format)
    else:
        if self.require_dense[0] and not enforce_sparse:
            # TODO: perhaps a check_array?
            return X
        else:
            return matrix_creation_function_for_format(sparse_format)(X)