import theano
from theano import tensor as tt
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import time

from utils import Utils
from regularizer import Regularizer

floatX = theano.config.floatX
rng = np.random.RandomState(123)
srng = RandomStreams(123)


class LogisticRegression(object):

    def __init__(self, X, distribution, fan_in, fan_out):

        self.weight = Weights(distribution)

        self.W = self.weight.init_weights(fan_in, fan_out,
                                          name='logistic.W')

        self.b = self.weight.init_weights(None, fan_out,
                                          name='logistic.b')

        self.util = Utils(X, self.W, self.b)

        self.p_y_given_x = self.util.softmax()
        # self.p_y_given_x = tt.nnet.softmax(tt.dot(X, self.W)+self.b)

        # self.y_pred = tt.argmax(self.p_y_given_x, axis=1, keepdims=True)
        # self.y_pred = tt.argmax(self.p_y_given_x, axis=1).astype(dtype=floatX)
        self.y_pred = tt.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        # self.X = X

    def neg_log_like(self, y):
        return self.util.neg_log_likelihood(y, self.p_y_given_x)

    def errors(self, y):
        return self.util.errors(y, self.y_pred)


class HiddenLayer(object):

    def __init__(self, X, distribution, fan_in, fan_out, activation=tt.tanh,
                 W=None, b=None):
        self.X = X
        if W is None:
            self.weight = Weights(distribution,
                                  low=-np.sqrt(6. / (fan_in + fan_out)),
                                  high=np.sqrt(6. / (fan_in + fan_out)))

            W = self.weight.init_weights(fan_in=fan_in,
                                         fan_out=fan_out,
                                         name='hidden.W')

            if activation == tt.nnet.sigmoid:
                W.set_value(W.get_value() * 4)

        if b is None:
            b = self.weight.init_weights(fan_out=fan_out, name='hidden.b')

        self.W = W
        self.b = b

        self.h_output = tt.dot(self.X, self.W) + self.b
        self.output = (
            self.h_output if activation is None
            else activation(self.h_output)
        )

        self.params = [self.W, self.b]


class MultiLayerPerceptron(object):

    def __init__(self, X, distribution, fan_in, n_hidden, fan_out):

        self.hiddenLayer = HiddenLayer(X, distribution, fan_in,
                                       fan_out=n_hidden)

        self.logRegressionLayer = LogisticRegression(
            self.hiddenLayer.h_output, distribution,
            fan_in=n_hidden, fan_out=fan_out
        )

        regularizer = Regularizer()

        self.L1_test = regularizer.lnorm([self.hiddenLayer.W,
                                         self.logRegressionLayer.W],
                                         'L1')

        self.L2_test = regularizer.lnorm([self.hiddenLayer.W,
                                         self.logRegressionLayer.W],
                                         'L2')
        self.L1 = (
            abs(self.hiddenLayer.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() +
            (self.logRegressionLayer.W ** 2).sum()
        )

        # the parameters of the model are the parameters of the two layer
        # it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.X = X

    def neg_log_like(self, y):
        return self.logRegressionLayer.neg_log_like(y)

    def errors(self, y):
        return self.logRegressionLayer.errors(y)


class AutoEncoder(object):

    def __init__(self, X, hidden_size, activation_function,
                 output_function):

        assert(len(X.get_value().shape) == 2)
        self.X = X
        self.m = X.get_value().shape[0]
        self.n = X.get_value().shape[1]
        self.output_function = output_function
        self.activation_function = activation_function
        # Hidden_size is the number of neurons in the hidden layer, an int.
        assert(type(hidden_size) is int)
        assert(hidden_size > 0)
        self.hidden_size = hidden_size
        weights = Weights(
            low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
            high=4 * np.sqrt(6. / (self.hidden_size + self.n))
        )

        self.W = weights.init_weights(fan_in=self.n,
                                      fan_out=self.hidden_size,
                                      name='AE.W')

        self.b1 = weights.weights_init(fan_out=self.hidden_size, name='AE.b1')
        self.b2 = weights.weights_init(fan_out=self.n, name='AE.b2')

    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = tt.lscalar()
        x = tt.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(tt.dot(x, self.W) + self.b1)
        output = tt.dot(hidden, tt.transpose(self.W)) + self.b2
        output = self.output_function(output)

        # Use cross-entropy loss.
        cost = -tt.mean(x * tt.log(output) + (1-x) * tt.log(1-output), axis=1)

        # Return gradient with respect to W, b1, b2.
        updates = []
        gparams = tt.grad(cost, params)

        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate * gparam))

        # Train given a mini-batch of the data.
        train = theano.function(inputs=[index],
                                outputs=[cost],
                                updates=updates,
                                givens={x:
                                        self.X[index: index +
                                               mini_batch_size, :]})

        tic = time.clock()
        for epoch in xrange(n_epochs):
            print "Epoch: ", epoch
            for row in xrange(0, self.m, mini_batch_size):
                train(row)
        toc = time.clock()
        print "Average time per epoch=", (toc - tic)/n_epochs

    def get_hidden(self, data):
        x = tt.fmatrix('x')
        hidden = self.activation_function(tt.dot(x, self.W) + self.b1)
        transformed_data = theano.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)

    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]


class DenoisyAutoEncoder(object):

    def __init__(self, rng=None, srgn=None, input=None, n_visible=784,
                 n_hidden=500, W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying:
        number of visible units (the dimension d of the input ),
        number of hidden units ( the dimension d' of latent/hidden space )
        corruption level.
        Such symbolic variables are useful when,
        for example the input is the result of some computations, or
        when weights are shared between the dA and an MLP layer. When
        dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type rng: np.random.RandomState
        :param rng: number random generator used to generate weights

        :type srgn: theano.tensor.shared_randomstreams.RandomStreams
        :param srgn: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not srgn:
            srgn = tt.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            weights = Weights(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible))
                )

            W = weights.init_weights(fan_in=n_visible,
                                     fan_out=n_hidden, name='DAE.W')

        if not bvis:
            bvis = weights.init_weights(fan_out=n_visible, name='DAE.bvis')

        if not bhid:
            bhid = weights.init_weights(fan_out=n_hidden, name='DAE.bhid')

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.srgn = srgn
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = tt.fmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        # return 1./(1 + tt.exp(-tt.dot(input, self.W) - self.b))
        return tt.nnet.sigmoid(tt.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """computes reconstructed input given the values of the
        hidden layer
        """
        return tt.nnet.sigmoid(tt.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
            """This function keeps ``1-corruption_level`` entries of
            the inputs the same and zero-out randomly selected subset
            of size ``coruption_level``
            Note : first argument of theano.rng.binomial is the shape(size) of
                   random numbers that it should produce
                   second argument is the number of trials
                   third argument is the probability of success of any trial

                    this will produce an array of 0s and 1s where 1 has a
                    probability of 1 - ``corruption_level`` and 0 with
                    ``corruption_level``

                    The binomial function return int64 data type by
                    default.  int64 multiplicated by the input
                    type(floatX) always return float64.  To keep all data
                    in floatX when floatX is float32, we set the dtype of
                    the binomial to floatX. As in our case the value of
                    the binomial is always 0 or 1, this don't change the
                    result. This is needed to allow the gpu to work
                    correctly as it only support float32 for now.

            """
            return self.srgn.binomial(size=input.shape,
                                      n=1,
                                      p=1 - corruption_level,
                                      dtype=floatX) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        """computes cost and updates for one trainng step of dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - tt.sum(self.x * tt.log(z) + (1 - self.x) * tt.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = tt.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = tt.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.params, gparams)]

        return (cost, updates)


class ContractiveAutoEncoder(object):
    """ Contractive Auto-Encoder class (cA)

    The contractive autoencoder tries to reconstruct the input with an
    additional constraint on the latent space. With the objective of
    obtaining a robust representation of the input space, we
    regularize the L2 norm(Froebenius) of the jacobian of the hidden
    representation with respect to the input. Please refer to Rifai et
    al.,2011 for more details.

    If x is the input then equation (1) computes the projection of the
    input into the latent space h. Equation (2) computes the jacobian
    of h with respect to x.  Equation (3) computes the reconstruction
    of the input, while equation (4) computes the reconstruction
    error and the added regularization term from Eq.(2).

    .. math::

        h_i = s(W_i x + b_i)                                             (1)

        J_i = h_i (1 - h_i) * W_i                                        (2)

        x' = s(W' h  + b')                                               (3)

        L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
             + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

    """

    def __init__(self, rng, input=None, n_visible=784, n_hidden=100,
                 n_batchsize=1, W=None, bhid=None, bvis=None):
        """Initialize the cA class by specifying the number of visible units
        (the dimension d of the input), the number of hidden units (the
        dimension d' of the latent or hidden space) and the contraction level.
        The constructor also receives symbolic variables for the input, weights
        and bias.

        :param rng: number random generator used to generate weights

        :param srng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :param input: a symbolic description of the input or None for
                      standalone cA

        :param n_visible: number of visible units

        :param n_hidden:  number of hidden units

        :param n_batchsize: number of examples per batch

        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            weights = Weights(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible))
                )

            W = weights.weights_init(fan_in=n_visible,
                                     an_out=n_hidden, name='CA.W')

        if not bvis:
            bvis = weights.weights_init(fan_out=n_visible, name='CA.bvis')

        if not bhid:
            bhid = weights.weights_init(fan_out=n_hidden, name='CA.bhid')

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = tt.fmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return tt.nnet.sigmoid(tt.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        reshape1 = tt.reshape(hidden * (1 - hidden),
                              (self.n_batchsize, 1, self.n_hidden))

        reshape2 = tt.reshape(W, (1, self.n_visible, self.n_hidden))

        return reshape1 * reshape2

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return tt.nnet.sigmoid(tt.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the cA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L_rec = - tt.sum(self.x * tt.log(z) +
                              (1 - self.x) * tt.log(1 - z), axis=1)

        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = tt.sum(J ** 2) // self.n_batchsize

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = tt.mean(self.L_rec) + contraction_level * tt.mean(self.L_jacob)

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        gparams = tt.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


class StackedDenoisyAutoEncoder(object):
    """Stacked denoising auto-encoder class (SdA)
    obtained stacking several dAs. The hidden layer of the dA at
    layer `i` becomes the input of the dA at layer `i+1`.
    The first layer dA gets as input is the input
    of the SdA, and the hidden layer of the last dA represents the output.
    After pretraining, the SdA is dealt as a normal MLP, the dAs are
    only used to initialize the weights.
    """

    def __init__(self, rng, srgn=None, n_ins=784,
                 hidden_layer_sizes=[500, 500], n_outs=10,
                 noise_levels=[0.1, 0.1]):
        """
        :param n_ins: dimension of the input to the sdA
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value
        :param n_outs: dimension of the output of the network
        :param noise_levels: amount of corruption to use for each
                                  layer
        """

        self.params = []
        self.n_layers = len(hidden_layer_sizes)
        self.sigmoid_layers = []
        self.dA_layers = []

        assert self.n_layers > 0

        if not srgn:
            srgn = RandomStreams(rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = tt.fmatrix('x')  # the data is presented as rasterized images
        self.y = tt.ivector('y')  # the labels are presented as 1D vector of
                                  # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        activation=tt.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = DenoisyAutoEncoder(rng=rng,
                                          srgn=srgn,
                                          input=layer_input,
                                          n_visible=input_size,
                                          n_hidden=hidden_layer_sizes[i],
                                          W=sigmoid_layer.W,
                                          bhid=sigmoid_layer.b)

            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients wrt the model parameters
        # symbolic variable, points to the number of errors made on the
        # minibatch, given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = tt.lscalar('index')  # index to a minibatch
        noise_level = tt.scalar('corruption')  # % of corruption to use
        learning_rate = tt.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(noise_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.In(noise_level, value=0.2),
                                         theano.In(learning_rate, value=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={
                                     self.x:
                                     train_set_x[batch_begin: batch_end]
                                 }
                            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = tt.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = tt.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [(param, param - gparam * learning_rate)
                   for param, gparam in zip(self.params, gparams)]

        train_fn = theano.function(inputs=[index], outputs=self.finetune_cost,
                                   updates=updates,
                                   givens={
                                       self.x:
                                       train_set_x[
                                           index * batch_size:
                                           (index + 1) * batch_size
                                           ],
                                       self.y: train_set_y[
                                           index * batch_size:
                                           (index + 1) * batch_size
                                           ]
                                    }, name='train'
                                )

        test_score_i = theano.function([index], self.errors,
                                       givens={
                                           self.x: test_set_x[
                                               index * batch_size:
                                               (index + 1) * batch_size
                                               ],
                                           self.y: test_set_y[
                                               index * batch_size:
                                               (index + 1) * batch_size
                                               ]
                                           }, name='test'
                                    )

        valid_score_i = theano.function([index], self.errors,
                                        givens={
                                            self.x: valid_set_x[
                                                index * batch_size:
                                                (index + 1) * batch_size
                                                ],
                                            self.y: valid_set_y[
                                                index * batch_size:
                                                (index + 1) * batch_size
                                                ]
                                            }, name='valid'
                                    )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


class RNN(object):
    pass


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None,
                 hbias=None, vbias=None, rng=None, srng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if srng is None:
            srng = RandomStreams(rng.randint(2 ** 30))

        if W is None:
            weights = Weights(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible))
                )

            W = weights.weights_init(fan_in=n_visible,
                                     fan_out=n_hidden,
                                     name='RBM.W')

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = weights.weights_init(fan_out=n_hidden, name='RBM.hbias')

        if vbias is None:
            # create shared variable for visible units bias
            vbias = weights.weights_init(fan_out=n_visible, name='RBM.vbias')

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = tt.fmatrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.srng = srng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = tt.dot(v_sample, self.W) + self.hbias
        vbias_term = tt.dot(v_sample, self.vbias)
        hidden_term = tt.sum(tt.log(1 + tt.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. Due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = tt.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation,
                tt.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that srng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.srng.binomial(size=h1_mean.shape,
                                       n=1,
                                       p=h1_mean,
                                       dtype=floatX)

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = tt.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation,
                tt.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that srng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.srng.binomial(size=v1_mean.shape,
                                       n=1,
                                       p=v1_mean,
                                       dtype=floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = tt.mean(self.free_energy(self.input)) - tt.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = tt.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * tt.cast(
                lr,
                dtype=floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = tt.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = tt.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = tt.mean(self.n_visible *
                       tt.log(tt.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  You need to understand a bit about how Theano works.
        Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = -tt.mean(
            self.input * tt.log(tt.nnet.sigmoid(pre_sigmoid_nv)) +
            (1 - self.input) * tt.log(1 - tt.nnet.sigmoid(pre_sigmoid_nv)),
            axis=1
            )

        return cross_entropy


# start-snippet-1
class DBN(object):
    """Deep Belief Network
    Obtained stacking several RBMs on top of each other.
    The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output.
    When used for classification, the DBN is treated as a MLP, by adding
    a logistic regression layer on top.
    """

    def __init__(self, rng, srng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :param rng: np random number generator used to draw initial
                    weights

        :param srng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :param n_ins: dimension of the input to the DBN

        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not srng:
            srng = MRG_RandomStreams(rng.randint(2 ** 30))

        # allocate symbolic variables for the data

        # the data is presented as rasterized images
        self.x = tt.fmatrix('x')

        # the labels are presented as 1D vector of [int] labels
        self.y = tt.ivector('y')
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=tt.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(rng=rng,
                            srng=srng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)

            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)

        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = tt.lscalar('index')  # index to a minibatch
        learning_rate = tt.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.In(learning_rate, value=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={
                                     self.x: train_set_x[batch_begin:batch_end]
                                 }
                            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = tt.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = tt.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index], outputs=self.finetune_cost,
                                   updates=updates,
                                   givens={
                                       self.x:
                                       train_set_x[index * batch_size:
                                                   (index + 1) * batch_size],
                                       self.y:
                                       train_set_y[index * batch_size:
                                                   (index + 1) * batch_size]
                                          }
                                   )

        test_score_i = theano.function([index], self.errors,
                                       givens={
                                           self.x:
                                           test_set_x[index * batch_size:
                                                      (index + 1) * batch_size],
                                           self.y:
                                           test_set_y[index * batch_size:
                                                      (index + 1) * batch_size]
                                              }
                                       )

        valid_score_i = theano.function([index], self.errors,
                                        givens={
                                            self.x:
                                            valid_set_x[index * batch_size:
                                                        (index + 1) * batch_size],
                                            self.y:
                                            valid_set_y[index * batch_size:
                                                        (index + 1) * batch_size]
                                               }
                                        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


class LSTM(object):
    pass


class ConvNet(object):
    pass


class AdversarialAutoEncoder(object):
    pass


class Weights(object):

    def __init__(self, distribution, low=0, high=1):

        self.low = low
        self.high = high

        switcher = {
            'uniform': rng.uniform,
            'binomial': rng.binomial,
            'normal': rng.normal,
            'beta': rng.beta
        }
        self.dist = switcher.get(distribution, rng.randn)

    def init_weights(self, fan_in=None, fan_out=None, name=None):
        try:
            if fan_in and fan_out:
                if self.dist == rng.randn:
                    params = self.dist(fan_in, fan_out).astype(dtype=floatX) * 0.01

                if self.dist == rng.uniform:
                    params = self.dist(low=self.low,
                                       high=self.high,
                                       size=(fan_in, fan_out)
                                       ).astype(dtype=floatX)

            elif fan_out and not fan_in:
                if self. dist == rng.uniform:
                    params = self.dist(low=self.low,
                                       high=self.high,
                                       size=(fan_out,)
                                       ).astype(dtype=floatX)

                if self.dist == rng.randn:
                    params = self.dist(fan_out).astype(dtype=floatX) * 0.01

            else:
                raise Exception()

            return theano.shared(value=params, name=name, borrow=True)

        except Exception:
                print("Check again your arguments, fan_in, fan_out " +
                      "and name")
