import theano
from theano import tensor as tt

import numpy as np

floatX = theano.config.floatX


class Utils(object):

    def __init__(self, X, W, b):
        self.X = X
        self.W = W
        self.b = b

    def softmax(self):
        dot_prod = tt.dot(self.X, self.W) + self.b
        numerator = tt.exp(dot_prod - tt.max(dot_prod, axis=1, keepdims=True))
        denominator = tt.sum(numerator, axis=1, keepdims=True)
        return numerator/denominator

    def neg_log_likelihood(self, y, p_y_given_x):
        if y.ndim > 1:
            return -tt.mean(tt.log(p_y_given_x)
                            [tt.arange(y.shape[0]), y.argmax(axis=1)])
        else:
            return -tt.mean(tt.log(p_y_given_x)
                            [tt.arange(y.shape[0]), y])
        # return tt.mean(tt.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def errors(self, y, y_pred):
        # if y.ndim != self.y_pred.ndim:
        #     raise UserWarning("Labels: y ==> {} should have the same "
        #                       "shape as the predicted labels: "
        #                       "self.y_pred ==> {}. Converting y "
        #                       "==> {} to self.y_pred ==> {}"
        #                       .format(y.type, self.y_pred.type,
        #                               y.type, y.argmax(axis=1).type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tt.mean(tt.neq(y_pred, y))
        elif y.dtype.startswith('float'):
            return tt.mean(tt.neq(y_pred, y.argmax(axis=1)),
                           dtype=floatX)
        else:
            raise NotImplementedError("errors() function not " +
                                      "implemented because labels " +
                                      " y.dtype are not implemented")

    def plot_first_k_numbers(self, W, k):
        from matplotlib import pyplot
        m = W.shape[0]
        k = min(m, k)
        j = int(round(k / 10.0))

        fig, ax = pyplot.subplots(j, 10)

        for i in range(k):

            w = W[i, :]

            w = w.reshape(28, 28)
            ax[i/10, i % 10].imshow(w, cmap=pyplot.cm.gist_yarg,
                                    interpolation='nearest', aspect='equal')

            ax[i/10, i % 10].axis('off')

        pyplot.tick_params(axis='x',  # changes apply to the x-axis
                           which='both',  # both major and minor ticks affected
                           bottom='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelbottom='off')
        pyplot.tick_params(axis='y',  # changes apply to the x-axis
                           which='both',  # both major and minor ticks affected
                           left='off',
                           right='off',  # ticks along the top edge are off
                           labelleft='off')

        fig.show()
