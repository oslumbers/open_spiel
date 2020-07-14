import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import linear_kernel

def kernel_solver(features):
    norm_features = preprocessing.normalize(features, norm='l2')
    return linear_kernel(norm_features)

def exp_cardinality(kernel):
    size = kernel.shape[0]
    inner_term = np.linalg.inv(kernel + np.eye(size))
    full_term = np.eye(size) - inner_term

    return np.trace(full_term)
