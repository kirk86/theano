import os
import sys
import tarfile
import urllib
import numpy as np
import theano

floatX = theano.config.floatX


class Loader(object):
    def __init__(self, datadir):
        self.datadir = datadir

    def one_hot(self, X, n):
        if type(X) == list:
            X = np.array(X)
        X = X.flatten()
        o_h = np.zeros((len(X), n))
        o_h[np.arange(len(X)), X] = 1
        return o_h.astype(dtype=floatX)

    def mnist(self, n_train=60000, n_test=10000, onehot=True):
        data_dir = os.path.join(self.datadir, 'mnist/')
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28*28)).astype(dtype=floatX)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        # trY = loaded[8:].reshape((60000)).astype(dtype=floatX)
        trY = loaded[8:].reshape((60000))

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28*28)).astype(dtype=floatX)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        # teY = loaded[8:].reshape((10000)).astype(dtype=floatX)
        teY = loaded[8:].reshape((10000))

        trX = trX/255.
        teX = teX/255.

        trX = trX[:n_train]
        trY = trY[:n_train]

        teX = teX[:n_test]
        teY = teY[:n_test]

        if onehot:
            trY = self.one_hot(trY, 10)
            teY = self.one_hot(teY, 10)
        else:
            trY = np.asarray(trY).astype(dtype=floatX)
            teY = np.asarray(teY).astype(dtype=floatX)

        return trX, trY, teX, teY

    def cifar10(self, onehot=True):
        import cPickle
        import re
        data_dir = os.path.join(self.datadir, 'cifar-10-batches-py/')
        files = os.listdir(data_dir)
        train_x = []
        train_y = []
        for filename in files:
            if re.search('_batch', filename) is not None:
                fh = open(data_dir + filename, 'rb')
                temp_data = cPickle.load(fh)
                if re.search('data_', filename) is not None:
                    train_x.append(temp_data['data'])
                    train_y.append(np.asarray(
                        temp_data['labels']
                        ).reshape(-1, 1))
                elif re.search('test_', filename) is not None:
                    teX = temp_data['data']
                    teY = np.asarray(temp_data['labels'])
                fh.close()

        counter = 0
        for tr_x in train_x:
            if counter <= len(train_x) - 1:
                if counter == 0:
                    trX = np.vstack((train_x[counter], train_x[counter + 1]))
                    trY = np.vstack((train_y[counter], train_y[counter + 1]))
                elif counter > 0 and counter < len(train_x) - 1:
                    trX = np.vstack((trX, train_x[counter + 1]))
                    trY = np.vstack((trY, train_y[counter + 1]))
                counter += 1

        del counter
        trX = trX/255.
        teX = teX/255.
        trY = trY.squeeze()

        if onehot:
            trY = self.one_hot(trY, 10)
            teY = self.one_hot(teY, 10)

        return trX, trY, teX, teY

    def stl10(self, onehot=True):
        # image shape
        HEIGHT = 96
        WIDTH = 96
        DEPTH = 3

        # size of a single image in bytes
        SIZE = HEIGHT * WIDTH * DEPTH

        # path to the directory with the data
        DATA_DIR = self.datadir

        # url of the binary data
        DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

        # path to the binary train file with image data
        TRAIN_DATA = DATA_DIR + 'stl10/train_X.bin'
        # path to the binary train file with labels
        TRAIN_LABEL = DATA_DIR + 'stl10/train_y.bin'

        TEST_DATA = DATA_DIR + 'stl10/test_X.bin'
        # path to the binary train file with labels
        TEST_LABEL = DATA_DIR + 'stl10/test_y.bin'

        UNLABELED = DATA_DIR + 'stl10/unlabeled_X.bin'

        # download data if needed
        # self.download_and_extract(DATA_DIR, DATA_URL)

        # test to check if the image is read correctly
        # with open(DATA_PATH) as f:
        #     image = read_single_image(f)
        #     plot_image(image)

        # test to check if the whole dataset is read correctly
        trX = self.read_all_images(TRAIN_DATA)
        trY = self.read_labels(TRAIN_LABEL)
        trY -= np.min(trY)

        teX = self.read_all_images(TEST_DATA)
        teY = self.read_labels(TEST_LABEL)
        teY -= np.min(teY)

        unlabeled = self.read_all_images(UNLABELED)

        trX = trX/255.
        teX = teX/255.

        if onehot:
            trY = self.one_hot(trY, 10)
            teY = self.one_hot(teY, 10)

        return trX, trY, teX, teY, unlabeled

    def read_labels(self, path_to_labels):
        """:param path_to_labels: path to the binary file containing labels
        from the STL-10 dataset :return: an array containing the
        labels
        """
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
        """
        :param path_to_data: the file containing the binary images from
        the STL-10 dataset :return: an array containing all the
        images
        """

        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, 96, 96, 3))

            for idx, image in zip(range(len(images)), images):
                images[idx] = np.rot90(image, 3)

            images = np.reshape(images, (-1, 3, 96, 96)).astype(dtype=floatX)

            # Now transpose the images into a standard image
            # format readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the
            # shuffle if you will use a learning algorithm like
            # CNN, since they like their channels separated.
            # images = np.transpose(images, (0, 3, 2, 1))
        return images

    def read_single_image(self, image_file, SIZE):
        """
        CAREFUL! - this method uses a file as input instead of
        the path - so the position of the reader will be
        remembered outside of context of this method.
        :param image_file: the open file containing the images
        :return: a single image
        """
        # read a single image, count determines the number of
        # uint8's to read
        image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
        # force into image matrix
        image = np.reshape(image, (3, 96, 96))
        # transpose to standard format You might want to
        # comment this line or reverse the shuffle if you will
        # use a learning algorithm like CNN, since they like
        # their channels separated.
        # image = np.transpose(image, (2, 1, 0))
        return image

    def download_and_extract(self, DATA_DIR, DATA_URL):
        """
        Download and extract the STL-10 dataset
        :return: None
        """
        dest_directory = DATA_DIR
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write("\rDownloading {}" +
                                 "{:.2f}%".format(filename,
                                                  float(count * block_size) /
                                                  float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
            print('Downloaded', filename)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def svhn(self):
        pass

    def ilsvrc2012_task1(self):
        pass

    def pascal_voc2007_comp3(self):
        pass

    def pascal_voc2007_comp4(self):
        pass

    def pascal_voc2010_comp3(self):
        pass

    def pascal_voc2010_comp4(self):
        pass

    def pascal_voc2011_comp3(self):
        pass

    def caltech_pedestrians(self):
        pass

    def inria_persons(self):
        pass

    def eth_pedestrians(self):
        pass

    def daimler_pedestrian(self):
        pass

    def tud_brussels_pedestrians(self):
        pass

    def kitti_vision_benchmark(self):
        pass

    def leeds_sport_poses(self):
        pass

    def msrc21(self):
        pass

    def salient_object_detection_benchmark(self):
        pass
