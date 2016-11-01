import theano
import theano.tensor as tt

import numpy as np


class Criterion(object):

    def __init__(self):
        pass

    def mse(self):
        pass

    def neg_log_likelihood(self):
        pass

    def cross_entropy(self, y, y_hat):
        """Also known as negative log likelihood or log-loss"""

        return -tt.mean(y * tt.log(y_hat) +
                        (1 - y) * tt.log(1 - y_hat), axis=1)
    # OR return -tt.mean(y*tt.log(y_hat)+(1-y)*tt.log(1-y_hat),axis=1)

    def max_margin(self):
        pass

    def auc_roc(self):
        """binary classification: useful for unbalanced test data
        1. http://stats.stackexchange.com/questions/132777/
        what-does-auc-stand-for-and-what-is-it

        2. http://mlwiki.org/index.php/ROC_Analysis"""
        pass

    def mathews_corr_coeff(self):
        pass

    def logloss(self):
        """ - 1/n \sum_{i=1}^{n} [ y_{i} \log(\hat{y}_{i}) +
        (1-y_{i}) \log(1-\hat{y}_{i}) ] """
        pass

    def f1_score(object):
        pass

    def lorenz_curves(object):
        pass

    def Wilcoxon_Mann_Whitney_Test(object):
        pass
