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

    def scale_to_unit_interval(self, ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def tile_raster_images(self, X, img_shape, tile_shape, tile_spacing=(0, 0),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.

        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).

        :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param X: a 2-D array in which every row is a flattened image.

        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image

        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)

        :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

        :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not


        :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
        :rtype: a 2-d array with same dtype as X.

        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape    = [0,0]
        # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [
            (ishp + tsp) * tshp - tsp
            for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]

        if isinstance(X, tuple):
            assert len(X) == 4
            # Create an output np.ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                        dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                        dtype=X.dtype)

            #colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    dt = out_array.dtype
                    if output_pixel_vals:
                        dt = 'uint8'
                    out_array[:, :, i] = np.zeros(
                        out_shape,
                        dtype=dt
                    ) + channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = self.tile_raster_images(
                        X[i], img_shape, tile_shape, tile_spacing,
                        scale_rows_to_unit_interval, output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

            # generate a matrix to store the output
            dt = X.dtype
            if output_pixel_vals:
                dt = 'uint8'
            out_array = np.zeros(out_shape, dtype=dt)

            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        this_x = X[tile_row * tile_shape[1] + tile_col]
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = self.scale_to_unit_interval(
                                this_x.reshape(img_shape))
                        else:
                            this_img = this_x.reshape(img_shape)
                        # add the slice to the corresponding position in the
                        # output array
                        c = 1
                        if output_pixel_vals:
                            c = 255
                        out_array[
                            tile_row * (H + Hs): tile_row * (H + Hs) + H,
                            tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
            return out_array
