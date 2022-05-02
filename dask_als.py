import numpy as np
from functools import reduce
import operator
import dask


def outer_product(*Xs):
    """ Outer/tensor product of multiple 2d arrays

    Parameters
    ----------
    *Xs: arrays of size (k, n)
        All inputs must have the same first dimension but may have varying
        second dimensions
    """
    n = len(Xs)
    indexes = [(slice(None, None),) +
               (None,) * i +
               (slice(None, None),) +
               (None,) * (n - i - 1) for i in range(len(Xs))]
    Ys = [X[ind] for X, ind in zip(Xs, indexes)]
    Ys = sorted(Ys, key=lambda y: y.nbytes)  # smaller outer products first
    return reduce(operator.mul, Ys)


import dask.array as da

def parafac_als(X, n_factors, n_iter=100):
    """ Parafac tensor decomposition

    This implements the basic algorithm in section 4.1 of this paper

        https://www.cs.cmu.edu/~pmuthuku/mlsp_page/lectures/Parafac.pdf
    """
    # Randomly initialize factors
    # factors = [np.random.random((n_factors, X.shape[i])) for i in range(0, X.ndim)]
    factors = [da.random.random((n_factors, X.shape[i]), 
                                chunks=(None, X.chunks[i])) 
               for i in range(0, X.ndim)]
    
    # Solve
    for itr in range(n_iter):
        for i in range(X.ndim):
            not_i = tuple(j for j in range(X.ndim) if j != i)
            Xp = X.transpose((i,) + not_i)
            Xp = Xp.reshape((Xp.shape[0], np.prod(Xp.shape[1:])))
            Z = outer_product(*[factors[j] for j in not_i])
            Z = Z.reshape((Z.shape[0], np.prod(Z.shape[1:])))

            # factor, residuals, rank, s = np.linalg.lstsq(Z.T, Xp.T)
            print(Z.T.shape, Xp.T.shape)
            factor, residuals, rank, s = da.linalg.lstsq(Z.T, Xp.T)
            factors[i] = factor
    return factors