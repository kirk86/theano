import theano
import theano.tensor as tt

import numpy as np
import timeit


class Trainer(object):

    def __init__(self):
        pass

    def train_logistic_regression(self):
        datasets = load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # allocate symbolic variables for the data
        index = tt.lscalar('index').astype(dtype='int32')  # index to a [mini]batch

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = tt.fmatrix('x').astype(dtype=floatX)  # data, presented as rasterized images
        y = tt.ivector('y').astype(dtype='int32')  # labels, presented as 1D vector of [int] labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = classifier.negative_log_likelihood(y)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                     givens={
                                         x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                     }
                                )

        validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                         givens={
                                             x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                             y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                         }
                                    )

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = tt.grad(cost=cost, wrt=classifier.W)
        g_b = tt.grad(cost=cost, wrt=classifier.b)

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                      givens={
                                          x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                     }
                                )
        # end-snippet-3

        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )

                        # save the best model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()

        print("Optimization complete with best validation score of {} %%,"
              "with test performance {} %%".format(best_validation_loss * 100.,
                                                   test_score * 100.)
             )

        print("The code run for {} epochs, with {} epochs/sec %".format(
                                epoch,
                                1. * epoch / (end_time - start_time)
                             )
             )

        print("The code for file {} ran for {:.1f}s.".format(
                                inspect.getfile(inspect.currentframe()),
                                (end_time - start_time)
                             )
              )


    def predict():
        """
        An example of how to load a trained model and use it
        to predict labels.
        """

        # load the saved model
        classifier = pickle.load(open('best_model.pkl'))

        # compile a predictor function
        predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

        # We can test it on some examples from test test
        dataset                = 'mnist.pkl.gz'
        datasets               = load_data(dataset)
        test_set_x, test_set_y = datasets[2]
        test_set_x             = test_set_x.get_value()

        predicted_values = predict_model(test_set_x[:10])
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

            print("Training epoch {}, cost {} for clear" +
                  " data".format(epoch, np.mean(cost_no_noise))

        toc = timeit.default_timer()

        training_time = (toc - tic)

        print("The no corruption code for file {} ran for " +
              "{:.2f}m".format(os.path.split("__file__")[1],
              (training_time / 60.)))

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
