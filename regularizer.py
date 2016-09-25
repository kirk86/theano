import theano
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

srng = RandomStreams(123)
rng = np.random.RandomState(123)

floatX = theano.config.floatX


class Regularizer(object):

    def __init__(self):
        pass

    def lnorm(self, params, method):
        if method == 'L1':
            for idx, param in zip(range(len(params)), params):
                params[idx] = abs(param).sum()
            return sum(params)
        if method == 'L2':
            for idx, param in zip(range(len(params)), params):
                params[idx] = (param ** 2).sum()
            return sum(params)

    def dropout(X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            # dropout nodes
            X *= srng.binomial(X.shape, p=retain_prob, dtype=floatX)

            X /= retain_prob  # scale the activations by (1-p). Looks like
                              # the inverted dropout only that in those
                              # examples scaling was performed by p instead
                              # of (1-p)
        return X

    def maxout(self):
        pass

    def elastinet(self):
        pass

    def lasso(self):
        pass

    def ridge(self):
        pass

    def akaike(self):
        pass

    def bayesian(self):
        pass
