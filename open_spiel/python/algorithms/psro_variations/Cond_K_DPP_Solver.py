from numpy.random import RandomState
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import linear_kernel

#from dppy.finite_dpps import FiniteDPP

def indexed_identity(indices, kernel):

    #Indices should be a list of the current population elements

    size = kernel.shape[0]
    I = np.eye(size)
    for idx in indices:
        I[idx, idx] = 0

    return I

def inner_term(kernel, current_popn):
    I = indexed_identity(current_popn, kernel)
    return np.linalg.inv(kernel + I)

def outer_term(indices, kernel):

    print(kernel)

    #Indices should be a list of the training population elements

    return np.linalg.inv(kernel[np.ix_(indices, indices)])

def conditional_k_dpp(meta_game, pop_size):

    total_popn = list(range(meta_game.shape[0]))
    current_popn = list(range(pop_size))
    training_popn = [x for x in total_popn if x not in current_popn]

    #normalise the meta_game
    meta_game_normalised = preprocessing.normalize(meta_game, norm='l2')

    #generate the L matrix
    L = linear_kernel(meta_game_normalised)
    print(L)

    #Generate the inner term
    L_inner = inner_term(L, current_popn)

    #Generate outer term
    L_outer = outer_term(training_popn, L_inner)

    #Return the final conditional K-DPP
    return L_outer - np.eye(L_outer.shape[0])
