import theano
import theano.tensor as tt

import numpy as np
import timeit
import inspect
import os

import cPickle as pickle

from models import LogisticRegression
from models import MultiLayerPerceptron
from models import AutoEncoder
from models import DenoisyAutoEncoder
from models import StackedDenoisyAutoEncoder
from utils import Utils

rng = np.random.RandomState(123)
srng = tt.shared_randomstreams.RandomStreams(np.random.randint(2 ** 30))


class Trainer(object):

    def __init__(self, dataset, n_epochs=50, batch_size=256, n_valid=10000):
        self.trX, self.trY, self.teX, self.teY = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # get last N elements as validation set
        if n_valid is not None:
            self.n_valid = n_valid
            self.mask = range(len(self.trX) - 1,
                              len(self.trX) - n_valid - 1,
                              -1)

            self.valX = self.trX[self.mask]
            self.valY = self.trY[self.mask]

            self.trX = np.delete(self.trX, self.mask, 0)
            self.trY = np.delete(self.trY, self.mask, 0)

            # compute number of minibatches for training,
            # validation and testing
            self.n_train_batch = self.trX.shape[0] // self.batch_size
            self.n_valid_batch = self.valX.shape[0] // self.batch_size
            self.n_test_batch = self.teX.shape[0] // self.batch_size

    def train_logistic_regression(self, distribution, fan_in, fan_out,
                                  learning_rate=0.013):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model.')

        X = tt.fmatrix('X')  # data, presented as rasterized images
        y = tt.fmatrix('y')  # labels, presented as 1-hot matrix of labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticRegression(X, distribution, fan_in, fan_out)

        # the cost we minimize during training
        cost = classifier.neg_log_like(y)

        # compute the gradient of cost with respect to theta = (W,b)
        grads = [tt.grad(cost=cost, wrt=param)
                 for param in classifier.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(classifier.params, grads)]

        # compiling a Theano function `train_model` that returns the cost,
        # in the same time updates the parameter based on the rules
        # defined in `updates`
        train = theano.function(inputs=[X, y],
                                outputs=cost,
                                updates=updates,
                                allow_input_downcast=True)

        # function that computes the mistakes that are made by
        # the model on a minibatch
        test = theano.function(inputs=[X, y],
                               outputs=classifier.errors(y),
                               allow_input_downcast=True)

        validate = theano.function(inputs=[X, y],
                                   outputs=classifier.errors(y),
                                   allow_input_downcast=True)

        predict = theano.function(inputs=[X], outputs=classifier.y_pred,
                                  allow_input_downcast=True)

        self.early_stopping(classifier, train, test, validate, predict,
                            learning_rate)

    def early_stopping(self, classifier, train, test, validate, predict=None,
                       learning_rate=0.013):

        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model.')
        # early-stopping parameters
        patience = 10 * self.n_train_batch  # look as this many examples
        patience_incr = 2  # wait this much longer when a new best is
                                      # found
        improv_thres = 0.995  # a relative improvement of this much is
                                      # considered significant
        valid_freq = min(self.n_train_batch, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_valid_error = np.inf
        # test_score = 0.
        tic = timeit.default_timer()

        done = False
        epoch = 0
        while (epoch < self.n_epochs) and (not done):
            epoch += 1
            for minibatch_idx in range(self.n_train_batch):

                minibatch_avg_cost = train(
                    self.trX[minibatch_idx * self.batch_size:
                             (minibatch_idx + 1) * self.batch_size],

                    self.trY[minibatch_idx * self.batch_size:
                             (minibatch_idx + 1) * self.batch_size]
                )

                train_cost = np.mean(minibatch_avg_cost)

                # iteration number
                iter_ = (epoch - 1) * self.n_train_batch + minibatch_idx

                # validate after each n_train_batch
                if (iter_ + 1) % valid_freq == 0:
                    # compute zero-one loss on validation set
                    for i in range(self.n_valid_batch):
                        valid_losses = [
                            validate(
                                self.valX[i * self.batch_size:
                                          (i + 1) * self.batch_size],

                                self.valY[i * self.batch_size:
                                          (i + 1) * self.batch_size]
                            )
                        ]

                    valid_error = np.mean(valid_losses)

                    print("epoch {}, minibatch {}/{},"
                          " train avg. cost per minibatch {},"
                          " validation errors {}%."
                          .format(epoch, minibatch_idx + 1,
                                  self.n_train_batch,
                                  train_cost,
                                  valid_error * 100.)
                          )

                    # if we got the best validation score until now
                    if valid_error < best_valid_error:
                        # improve patience if loss improvement is good enough
                        if valid_error < best_valid_error * improv_thres:
                            patience = max(patience, iter_ * patience_incr)

                        best_valid_error = valid_error

                        # test it on the test set
                        for i in range(self.n_test_batch):
                            test_losses = [
                                test(
                                    self.teX[i * self.batch_size:
                                             (i + 1) * self.batch_size],
                                    self.teY[i * self.batch_size:
                                             (i + 1) * self.batch_size]
                                )
                            ]

                        test_error = np.mean(test_losses)

                        print("epoch {}, minibatch {}/{}, test error of"
                              " best model {}%."
                              .format(epoch, minibatch_idx + 1,
                                      self.n_train_batch, test_error * 100.))

                        # save the best model
                        if os.path.isdir('log_regress') is False:
                            os.mkdir('log_regress')
                            os.chdir('log_regress')
                            with open('best_model.pkl', 'wb') as f:
                                pickle.dump(classifier, f)

                if patience <= iter_:
                    done = True
                    break

        toc = timeit.default_timer()

        print("Optimization complete with best validation error of {}%,"
              " with test error {}%."
              .format(best_valid_error * 100., test_error * 100.))

        print("Model accuracy on test set: {}%."
              .format(
                  np.mean(
                      np.argmax(self.teY, axis=1) == predict(self.teX)
                         ) * 100
                     )
              )

        print("The code run for {} epochs, with {} epochs/sec."
              .format(epoch, 1. * epoch / (toc - tic)))

        print("The code for file {} ran for {:.1f}s."
              .format(inspect.getfile(inspect.currentframe()),
                      (toc - tic)))

    def predict(self):
        """
        An example of how to load a trained model and use it
        to predict labels.
        """

        # load the saved model
        classifier = pickle.load(open('best_model.pkl'))

        # compile a predictor function
        predict = theano.function(inputs=[X],
                                  outputs=classifier.y_pred)

        # We can test it on some examples from test test
        # dataset                = 'mnist.pkl.gz'
        # datasets               = load_data(dataset)
        # test_set_x, test_set_y = datasets[2]
        # test_set_x             = test_set_x.get_value()

        predicted_values = predict(self.teX[:10])
        print("Predicted values for the first 10 examples in test set:")
        print(predicted_values)

    def train_multilayer_perceptron(self, distribution, fan_in,
                                    n_hidden, fan_out, learning_rate=0.012):

        X = tt.fmatrix('X')
        y = tt.fmatrix('y')

        classifier = MultiLayerPerceptron(X, distribution, fan_in,
                                          n_hidden, fan_out)

        cost = (
            classifier.neg_log_like(y) +
            0.00 * classifier.L1 +
            0.0001 * classifier.L2_sqr
            )

        gparams = [tt.grad(cost=cost, wrt=param)
                   for param in classifier.params]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        train = theano.function(inputs=[X, y], outputs=cost,
                                updates=updates,
                                allow_input_downcast=True)

        test = theano.function(inputs=[X, y],
                               outputs=classifier.errors(y),
                               allow_input_downcast=True)

        validate = theano.function(inputs=[X, y],
                                   outputs=classifier.errors(y),
                                   allow_input_downcast=True)

        predict = theano.function(inputs=[X],
                                  outputs=classifier.logRegressionLayer.y_pred,
                                  allow_input_downcast=True)

        self.early_stopping(classifier, train, test, validate, predict)

    def train_autoencoder(self, distribution, fan_in, fan_out,
                          learning_rate=0.01):
        # allocate symbolic variables for the data
        X = tt.fmatrix('X')  # the data is presented as rasterized images

        autoEncoder = AutoEncoder(X, distribution, fan_in, fan_out,
                                  n_hidden=500, activation=tt.nnet.sigmoid,
                                  output=tt.nnet.sigmoid)

        gparams = [tt.grad(cost=autoEncoder.cost, wrt=param)
                   for param in autoEncoder.params]

        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(autoEncoder.params, gparams)]

        train = theano.function(inputs=[X], outputs=autoEncoder.cost,
                                updates=updates,
                                allow_input_downcast=True)

        ############
        # TRAINING #
        ############

        tic = timeit.default_timer()

        # go through training epochs
        for epoch in range(self.n_epochs):
            # go through trainng set
            for batch_idx in range(0, self.n_train_batch, self.batch_size):
                train_loss = train(self.trX[batch_idx: batch_idx + self.batch_size])

            print("Training epoch {}, cost {}. %%"
                  .format(epoch + 1, train_loss))

        toc = timeit.default_timer()
        training_time = (toc - tic)
        print("Average time per epoch = {}.%%".format(training_time))

        # plot encoding weights
        weight = [param
                  for param in autoEncoder.get_params()
                  if param.shape >= 2]

        util = Utils(None, None, None)
        # util.plot_first_k_numbers(weight[0], 100)
        # plot decoding weights
        util.plot_first_k_numbers(np.transpose(weight[0]), 100)

        # start-snippet-4
        # image = Image.fromarray(
        #     tile_raster_images(X=dae.W.get_value(borrow=True).T,
        #                        img_shape=(28, 28), tile_shape=(10, 10),
        #                        tile_spacing=(1, 1)))
        # image.save('filters_no_corruption.png')
        # # end-snippet-4

    def train_denoisy_autoencoder(self, distribution, n_visible, n_hidden):

        X = tt.fmatrix('X')  # the data is presented as rasterized images

        #####################################
        # BUILDING THE MODEL CORRUPTION 0% #
        #####################################

        dae = DenoisyAutoEncoder(X, distribution, n_visible, n_hidden)

        cost, updates = dae.get_cost_updates(0., 0.01)

        train = theano.function([X], cost, updates=updates,
                                allow_input_downcast=True)

        ############
        # TRAINING #
        ############

        tic = timeit.default_timer()

        # go through training epochs
        for epoch in range(self.n_epochs):
            # go through trainng set
            cost_noiseless = []
            for batch_idx in range(self.n_train_batch):
                cost_noiseless.append(
                    train(self.trX[batch_idx * self.batch_size:
                                   (batch_idx + 1) * self.batch_size])
                )

            print("Training epoch {}, cost for clear data {} %"
                  .format(epoch + 1, np.mean(cost_noiseless)))

        toc = timeit.default_timer()

        train_duration = (toc - tic)

        print("The no corruption code for file {} ran for {:.2f}m %"
              .format(inspect.getfile(inspect.currentframe()),
                      (train_duration / 60.)))

        # image = Image.fromarray(
        # tile_raster_images(X=dae.W.get_value(borrow=True).T,
        #     img_shape=(28, 28), tile_shape=(10, 10),
        #     tile_spacing=(1, 1)))
        # image.save('filters_no_corruption.png')

        #####################################
        # BUILDING THE MODEL CORRUPTION 30% #
        #####################################

        dae_noisy = DenoisyAutoEncoder(X, distribution, n_visible, n_hidden)

        cost_noisy, updates_noisy = dae_noisy.get_cost_updates(0.3, 0.01)

        train_noisy = theano.function([X], cost_noisy, updates=updates_noisy,
                                      allow_input_downcast=True)

        ############
        # TRAINING #
        ############

        tic = timeit.default_timer()

        # go through training epochs
        for epoch in range(self.n_epochs):
            # go through trainng set
            cost_noisy = []
            for batch_idx in range(self.n_train_batch):
                cost_noisy.append(
                    train_noisy(self.trX[batch_idx * self.batch_size:
                                         (batch_idx + 1) * self.batch_size])
                )

            print("Training epoch {}, cost for noisy data {} %"
                  .format(epoch + 1, np.mean(cost_noisy)))

        toc = timeit.default_timer()

        train_duration = (toc - tic)

        print("The 30% corruption code for file {}, ran for {:.2f}m %"
              .format(inspect.getfile(inspect.currentframe()),
                      (train_duration / 60.)))

        # image = Image.fromarray(
        # tile_raster_images(X=dae_noise.W.get_value(borrow=True).T,
        #     img_shape=(28, 28), tile_shape=(10, 10),
        #     tile_spacing=(1, 1)))
        # image.save('filters_corruption_30.png')

    def train_stacked_denoisy_autoencoder(self, distribution):

        X = tt.fmatrix('X')
        y = tt.fmatrix('y')

        print('... building the model')
        # construct the stacked denoising autoencoder class

        #########################
        # PRETRAINING THE MODEL #
        #########################
        print('... getting the pretraining functions')
        index            = tt.lscalar('index')  # index to a minibatch
        corruption_level = tt.scalar('corruption')  # % of corruption to use
        learning_rate    = tt.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin      = index * batch_size
        # ending of a batch given `index`
        batch_end        = batch_begin + batch_size

        pretrain_fcns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fcn = theano.function(inputs=[X,
                                         theano.In(corruption_level, value=0.2),
                                         theano.In(learning_rate, value=0.1)],
                                  outputs=cost,
                                  updates=updates, allow_input_downcast=True)
            pretrain_fcns.append(fcn)

        print('... pre-training the model')
        tic = timeit.default_timer()
        # Pre-train layer-wise
        noise_levels = [.1, .2, .3]
        for i in range(sda.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                loss = []
                for batch_index in range(self.n_train_batch):
                    loss.append(pretraining_fcns[i](index=batch_index,
                                                    corruption=corruption_levels[i],
                                                    lr=pretrain_lr))
                print("Pre-training layer {}, epoch {}, cost {}"
                      .format(i + 1, epoch, np.mean(loss)))

        toc = timeit.default_timer()

        print("The pretraining code for file {} ran for {:.2f}m"
              .format((toc - tic)/60.,
                      inspect.getfile(inspect.currentframe())))
        ########################
        # FINETUNING THE MODEL #
        ########################
        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')

        index = tt.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = tt.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [(param, param - gparam * learning_rate)
                   for param, gparam in zip(self.params, gparams)]

        train = theano.function(inputs=[X, y], outputs=self.finetune_cost,
                                updates=updates, allow_input_downcast=True)

        test = theano.function([X, y], self.errors, allow_input_downcast=True)

        valid = theano.function([X, y], self.errors, allow_input_downcast=True)

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test(i) for i in range(n_test_batches)]

        print("... finetuning the model")
        self.early_stopping(train, test, valid, predict=None)
