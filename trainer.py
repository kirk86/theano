import theano
import theano.tensor as tt

import numpy as np
import timeit
import inspect
import os

import cPickle as pickle

from models import LogisticRegression


class Trainer(object):

    def __init__(self, dataset, n_epochs=1000, batch_size=950, N=10000):
        self.trX, self.trY, self.teX, self.teY = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # get last N elements as validation set
        if N is not None:
            self.N = N
            self.mask = range(len(self.trX) - 1, len(self.trX) - N - 1, -1)

            self.valX = self.trX[self.mask]
            self.valY = self.trY[self.mask]

            self.trX = np.delete(self.trX, self.mask, 0)
            self.trY = np.delete(self.trY, self.mask, 0)

    def train_logistic_regression(self, fan_in, fan_out, learning_rate=0.013):

        # compute number of minibatches for training, validation and testing
        n_train_batch = self.trX.shape[0] // self.batch_size
        n_valid_batch = self.valX.shape[0] // self.batch_size
        n_test_batch = self.teX.shape[0] // self.batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model.')

        # index = tt.lscalar('index')  # index to a [mini]batch

        X = tt.fmatrix('X')  # data, presented as rasterized images
        y = tt.fmatrix('y')  # labels, presented as 1-hot matrix of labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticRegression(X, fan_in, fan_out)

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

        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model.')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_incr = 2  # wait this much longer when a new best is
                                      # found
        improv_thres = 0.995  # a relative improvement of this much is
                                      # considered significant
        valid_freq = min(n_train_batch, patience // 2)
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
            for minibatch_idx in range(n_train_batch):

                minibatch_avg_cost = train(
                    self.trX[minibatch_idx * self.batch_size:
                             (minibatch_idx + 1) * self.batch_size],

                    self.trY[minibatch_idx * self.batch_size:
                             (minibatch_idx + 1) * self.batch_size]
                )

                train_cost = np.mean(minibatch_avg_cost)

                # iteration number
                iter_ = (epoch - 1) * n_train_batch + minibatch_idx

                # validate after each n_train_batch
                if (iter_ + 1) % valid_freq == 0:
                    # compute zero-one loss on validation set
                    for i in range(n_valid_batch):
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
                                  n_train_batch,
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
                        for i in range(n_test_batch):
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
                                      n_train_batch, test_error * 100.))

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

    def train_multilayer_perceptron(self):
        pass

    def train_autoencoder(self, n_epochs, noise):
        ############
        # TRAINING #
        ############

        tic = timeit.default_timer()

        # go through training epochs
        for epoch in range(training_epoch):
            # go through trainng set
            cost_no_noise = []
            for batch_index in range(n_train_batches):
                cost_no_noise.append(train_dae(batch_index))

            print("Training epoch {}, cost {} for clear  data"
                  .format(epoch, np.mean(cost_no_noise)))

        toc = timeit.default_timer()

        training_time = (toc - tic)

        print("The no corruption code for file {} ran for {:.2f}m"
              .format(os.path.split("__file__")[1], (training_time / 60.)))

        # start-snippet-4
        image = Image.fromarray(
            tile_raster_images(X=dae.W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('filters_no_corruption.png')
        # end-snippet-4


# def test_dae():
#     # allocate symbolic variables for the data
#     index = tt.lscalar() # index to a [mini]batch
#     x = tt.matrix('x')  # the data is presented as rasterized images

#     rng  = np.random.RandomState(123)
#     srgn = tt.shared_randomstreams.RandomStreams(np.random.randint(2 ** 30))

#     data = load_data('mnist')
#     trX, trY = data[0]
#     batch_size = 20
#     training_epoch = 15
#     learning_rate = 0.1
#     n_train_batches = trX.get_value(borrow=True).shape[0] // batch_size

#     # start-snippet-2
#     #####################################
#     # BUILDING THE MODEL CORRUPTION 0% #
#     #####################################

#     dae = DAE(
#         rng=rng,
#         srgn=srgn,
#         input=x,
#         n_visible=28 * 28,
#         n_hidden=500
#     )

#     cost, updates = dae.get_cost_updates(
#                 corruption_level=0.,
#                 learning_rate=0.01
#     )

#     train_dae = theano.function([index], cost, updates=updates,
#             givens={x: trX[index * batch_size: (index + 1) * batch_size]}
#     )

#     ############
#     # TRAINING #
#     ############

#     tic = timeit.default_timer()

#     # go through training epochs
#     for epoch in range(training_epoch):
#         # go through trainng set
#         cost_no_noise = []
#         for batch_index in range(n_train_batches):
#             cost_no_noise.append(train_dae(batch_index))

#         print('Training epoch %d, cost for clear data' % epoch, np.mean(cost_no_noise))

#     toc = timeit.default_timer()

#     training_time = (toc - tic)

#     print(('The no corruption code for file ' +
#            os.path.split(__file__)[1] +
#            ' ran for %.2fm' % (training_time / 60.)))

#     # start-snippet-4
#     image = Image.fromarray(
#     tile_raster_images(X=dae.W.get_value(borrow=True).T,
#         img_shape=(28, 28), tile_shape=(10, 10),
#         tile_spacing=(1, 1)))
#     image.save('filters_no_corruption.png')
#     # end-snippet-4


#     # start-snippet-3
#     #####################################
#     # BUILDING THE MODEL CORRUPTION 30% #
#     #####################################

#     dae_noise = DAE(
#         rng=rng,
#         srgn=srgn,
#         input=x,
#         n_visible=28 * 28,
#         n_hidden=500
#     )

#     cost_noise, updates_noise = dae_noise.get_cost_updates(
#         corruption_level=0.3,
#         learning_rate=0.01
#     )

#     train_noise_dae = theano.function([index], cost_noise, updates=updates_noise,
#         givens={x: trX[index * batch_size: (index + 1) * batch_size]}
#     )

#     ############
#     # TRAINING #
#     ############

#     tic = timeit.default_timer()

#     # go through training epochs
#     for epoch in range(training_epoch):
#         # go through trainng set
#         cost_noise = []
#         for batch_index in range(n_train_batches):
#             cost_noise.append(train_dae(batch_index))

#         print('Training epoch %d, cost for clear data' % epoch, np.mean(cost_noise))

#     toc = timeit.default_timer()

#     training_time = (toc - tic)

#     print(('The 30% corruption code for file ' +
#            os.path.split(__file__)[1] +
#            ' ran for %.2fm' % (training_time / 60.)))

#     # start-snippet-4
#     image = Image.fromarray(
#     tile_raster_images(X=dae_noise.W.get_value(borrow=True).T,
#         img_shape=(28, 28), tile_shape=(10, 10),
#         tile_spacing=(1, 1)))
#     image.save('filters_corruption_30.png')
#     # end-snippet-4


# if __name__ == '__main__':
#     test_dae()
