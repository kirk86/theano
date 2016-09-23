import theano
import theano.tensor as tt

import numpy as np

floatX = theano.config.floatX


class Optimizer(object):

    def __init__(self):
        pass

    def stochastic_gradient_descent(self):
        pass

    def conjugate_gradient(self):
        pass

    def adam(cost, params, learning_rate=0.0002, beta1=0.1,
             beta2=0.001, epsilon=1e-8, gamma=1-1e-7):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        :parameters:
            - cost : Theano expression specifying loss/cost function
            - params : list of theano.tensors
                Gradients are calculated w.r.t. tensors in params
            - learning_Rate : float
            - beta1 : float
                Exponentioal decay rate on 1. moment of gradients
            - beta2 : float
                Exponentioal decay rate on 2. moment of gradients
            - epsilon : float
                For numerical stability
            - gamma: float
                Decay on first moment running average coefficient
            - Returns: list of update rules
        """

        updates = []
        grads = theano.grad(cost, params)

        i = theano.shared(np.float32(1))  # scalar shared
        i_t = i + 1.
        beta1_t = 1. - (1. - beta1) * gamma ** (i_t - 1)   # ADDED

        learning_rate_t = learning_rate * (tt.sqrt(1. - (1. - beta2) ** i_t) /
                                           1. - (1. - beta1) ** i_t)

        for p, g in zip(params, grads):
            m = theano.shared(np.zeros(p.get_value().shape, dtype=floatX))
            v = theano.shared(np.zeros(p.get_value().shape, dtype=floatX))

            m_t = (beta1_t * g) + ((1. - beta1_t) * m)  # CHANGED from
                                                        # b_t to beta1_t
            v_t = (beta2 * g**2) + ((1. - beta2) * v)

            g_t = m_t / (tt.sqrt(v_t) + epsilon)

            p_t = p - (learning_rate_t * g_t)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

    def adadelta(self):
        pass

    def rmsprop(cost, params, step_size=0.001, rho=0.9, epsilon=1e-6):
        grads = tt.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            g = g / tt.sqrt(acc_new + epsilon)  # gradient scaling
            updates.append((acc, acc_new))
            updates.append((p, p - step_size * g))
        return updates

    def nesterov_accelerted_gradient(self):
        pass
