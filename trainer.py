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
from models import ContractiveAutoEncoder
from models import RBM
from utils import Utils

from PIL import Image

rng = np.random.RandomState(123)
srng = tt.shared_randomstreams.RandomStreams(np.random.randint(2 ** 30))
theano.config.exception_verbosity = 'high'

floatX = theano.config.floatX


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

        if predict is not None:
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

    def predict(self, X):
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

        print("The 30% corruption code for file {}, ran for {:.2f}m %"
              .format(inspect.getfile(inspect.currentframe()),
                      ((toc - tic) / 60.)))

        # image = Image.fromarray(
        # tile_raster_images(X=dae_noise.W.get_value(borrow=True).T,
        #     img_shape=(28, 28), tile_shape=(10, 10),
        #     tile_spacing=(1, 1)))
        # image.save('filters_corruption_30.png')

    def train_stacked_denoisy_autoencoder(self, distribution,
                                          batch_size=128,
                                          pretrain_epochs=5,
                                          pretrain_lr=0.01,
                                          finetune_lr=0.01):

        print('... building the model')
        # construct the stacked denoising autoencoder class
        sda = StackedDenoisyAutoEncoder(distribution,
                                        fan_in=784,
                                        n_hidden_sizes=[500, 1024, 2048],
                                        fan_out=10,
                                        noise_levels=[0.1, 0.2, 0.3])

        #########################
        # PRETRAINING THE MODEL #
        #########################
        print('... getting the pretraining functions')
        pretrain_fcns = sda.pretrain_fcns(self.batch_size)

        print('... pre-training the model')
        tic = timeit.default_timer()
        # Pre-train layer-wise
        noise_levels = [.1, .2, .3]
        for i in range(sda.n_layers):
            # go through pretraining epochs
            for epoch in range(pretrain_epochs):
                # go through the training set
                loss = []
                for batch_idx in range(self.n_train_batch):
                    loss.append(pretrain_fcns[i]
                                (self.trX[batch_idx * self.batch_size:
                                          (batch_idx + 1) * self.batch_size],
                                 noise_levels[i],
                                 pretrain_lr))
                print("Pre-training layer {}, epoch {}, cost {}"
                      .format(i + 1, epoch + 1, np.mean(loss)))

        toc = timeit.default_timer()

        print("The pretraining code for file {} ran for {:.2f}m"
              .format(inspect.getfile(inspect.currentframe()),
                      (toc - tic)/60.))
        ########################
        # FINETUNING THE MODEL #
        ########################
        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')

        dataset = [(self.trX, self.trY),
                   (self.valX, self.valY),
                   (self.teX, self.teY)]

        train, valid, test = sda.finetune_fcns(dataset,
                                               self.batch_size,
                                               finetune_lr,
                                               self.n_valid_batch,
                                               self.n_train_batch)
        print("... finetuning the model")
        self.early_stopping(sda, train, test, valid)

    def train_contractive_autoencoder(self, distribution, batch_size=128,
                                      contraction_level=.1,
                                      learning_rate=0.01,
                                      output_folder='cA_plots'):

        # allocate symbolic variables for the data
        X = tt.fmatrix('X')  # the data is presented as rasterized images

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
        ####################################
        #        BUILDING THE MODEL        #
        ####################################

        contractive_autoencoder = ContractiveAutoEncoder(distribution, input=X,
                                                         n_visible=28*28,
                                                         n_hidden=500,
                                                         n_batchsize=self.batch_size)

        cost, updates = contractive_autoencoder.get_cost_updates(
            contraction_level=contraction_level,
            learning_rate=learning_rate)

        train = theano.function([X], [tt.mean(contractive_autoencoder.L_rec),
                                      contractive_autoencoder.L_jacob],
                                updates=updates)

        tic = timeit.default_timer()

        ############
        # TRAINING #
        ############

        # go through training epochs
        for epoch in range(self.n_epochs):
            # go through trainng set
            cost = []
            for batch_idx in range(self.n_train_batch):
                cost.append(train(self.trX[batch_idx * self.batch_size:
                                           (batch_idx + 1) * self.batch_size]))

            c_array = np.vstack(cost)
            print("Training epoch {}, reconstruction cost {},"
                  " jacobian norm {}".format(epoch, np.mean(
                      c_array[0]), np.mean(np.sqrt(c_array[1]))))

        toc = timeit.default_timer()

        print("The code for file {} ran for {:.2f}m."
              .format(inspect.getfile(inspect.currentframe()),
                      (toc - tic)/60.))
        try:
            import PIL.Image as Image
        except ImportError:
            import Image

        image = Image.fromarray(Utils.tile_raster_images(
            X=contractive_autoencoder.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))

        image.save('cae_filters.png')

        os.chdir('../')

    def train_rbm(self, distribution, batch_size=20,
                  learning_rate=0.1, training_epochs=15, n_chains=20,
                  n_samples=10, output_folder='rbm_plots',
                  n_hidden=500):
        """
        :param batch_size: size of a batch used to train the RBM

        :param n_chains: number of parallel Gibbs chains to be used
        for sampling

        :param n_samples: number of samples to plot for each chain

        """

        # allocate symbolic variables for the data
        X = tt.fmatrix('X')  # the data is presented as rasterized images

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                                  dtype=floatX),
                                         borrow=True)

        # construct the RBM class
        rbm = RBM(distribution, input=X, n_visible=28 * 28,
                  n_hidden=n_hidden, rng=rng, srng=srng)

        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                             persistent=persistent_chain, k=15)

        #################################
        #     Training the RBM          #
        #################################
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)

        # start-snippet-5
        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train = theano.function([X], [cost], updates=updates, name='train_rbm')

        plotting_time = 0.
        tic = timeit.default_timer()

        # go through training epochs
        for epoch in range(training_epochs):

            # go through the training set
            mean_cost = []
            for batch_idx in range(self.n_train_batch):
                mean_cost += [train(self.trX[batch_idx * batch_size:
                                             (batch_idx + 1) * batch_size])]

            print("Training epoch {}, cost is {}"
                  .format(epoch, np.mean(mean_cost)))

            # Plot filters after each training epoch
            plotting_start = timeit.default_timer()
            # Construct image from the weight matrix
            util = Utils(X, rbm.W, rbm.vbias)
            image = Image.fromarray(
                util.tile_raster_images(X=rbm.W.get_value(borrow=True).T,
                                        img_shape=(28, 28),
                                        tile_shape=(10, 10),
                                        tile_spacing=(1, 1)))

            image.save('filters_at_epoch_%i.png' % epoch)
            plotting_stop = timeit.default_timer()
            plotting_time += (plotting_stop - plotting_start)

        toc = timeit.default_timer()

        pretraining_time = (toc - tic) - plotting_time

        print ("Training took {} minutes".format(pretraining_time / 60.))
        # end-snippet-5 start-snippet-6
        #################################
        #     Sampling from the RBM     #
        #################################
        # find out the number of test samples
        number_of_test_samples = self.teX.shape[0]

        # pick random test examples, with which to initialize the
        # persistent chain
        test_idx = rng.randint(number_of_test_samples - n_chains)
        persistent_vis_chain = theano.shared(
            np.asarray(self.teX[test_idx:test_idx + n_chains],
                       dtype=floatX)
        )
        # end-snippet-6 start-snippet-7
        plot_every = 1000
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `plot_every` steps before returning the
        # sample for plotting
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            rbm.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every,
            name="gibbs_vhv"
        )

        # add to updates the shared variable that takes care of our persistent
        # chain :.
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function([], [vis_mfs[-1],
                                         vis_samples[-1]],
                                    updates=updates,
                                    name='sample_fn')

        # create a space to store the image for plotting ( we need to leave
        # room for the tile_spacing as well)
        image_data = np.zeros(
            (29 * n_samples + 1, 29 * n_chains - 1),
            dtype='uint8'
        )
        for idx in range(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()
            print(" ... plotting sample {}".format(idx))
            image_data[29 * idx:29 * idx + 28, :] = util.tile_raster_images(
                X=vis_mf,
                img_shape=(28, 28),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )

        # construct image
        image = Image.fromarray(image_data)
        image.save('samples.png')
        # end-snippet-7
        os.chdir('../')
