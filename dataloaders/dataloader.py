import threading
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy.io import loadmat
import os


class DataLoader:
    """Images are returned in 0-1 normalization"""
    def __init__(self, batch_size):
        self.train_dir = None
        self.validation_dir = None
        self.test_dir = None
        self.list_attrib_file = None
        self.train_smpls = None
        self.valid_smpls = None
        self.test_smpls = None
        self.labels = None
        self.data_dir = None

        self.datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                          samplewise_center=False,  # set each sample mean to 0
                                          featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                          samplewise_std_normalization=False,  # divide each input by its std
                                          zca_whitening=False,  # apply ZCA whitening
                                          rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                          width_shift_range=0.0,  # randomly shift horizontally (fraction of tot width)
                                          height_shift_range=0.0,  # randomly shift vertically (fraction of tot height)
                                          horizontal_flip=False,  # randomly flip images
                                          vertical_flip=False)
        self.batch_size = batch_size

    def _load_labels(self):
        if self.labels is not None:
            return
        if self.train_smpls is None:
            raise ValueError('Call get_data_loader first so train, test and validation directories are set.')
        data_root_dir, _ = os.path.split(self.train_dir)
        self.labels = np.genfromtxt(self.list_attrib_file, delimiter=',', skip_header=1, usecols=list(range(39))[1:])

    def get_data_dir(self):
        if self.data_dir is None:
            raise ValueError('Root directory for the data set was not specified!')
        return self.data_dir

    def get_train_labels(self):
        self._load_labels()
        return self.labels[:self.train_smpls]

    def get_validation_labels(self):
        self._load_labels()
        return self.labels[self.train_smpls:self.train_smpls + self.valid_smpls]

    def get_test_labels(self):
        if self.test_smpls is None:
            return None
        self._load_labels()
        return self.labels[self.train_smpls + self.valid_smpls:self.train_smpls + self.valid_smpls + self.test_smpls]

    def make_auto_encoder_gen_frm_keras_image_gen(self, generator):
        batch_size = self.batch_size
        class ThreadSafeIterator:

            def __init__(self, generator):
                self.lock = threading.Lock()
                self.generator = generator

            def next(self):
                with self.lock:
                    x = self.generator.next()
                    if x.shape[0] != batch_size:
                        x = self.generator.next()
                    x = x.astype(np.float32)/255.0
                    return x, x

        return ThreadSafeIterator(generator)

    def get_generator_from_img_dir(self, sorce_dir, classes, img_size, shuffle):
        generator = self.datagen.flow_from_directory(sorce_dir,
                                                     batch_size=self.batch_size,
                                                     class_mode=None,
                                                     classes=classes,
                                                     target_size=img_size,
                                                     shuffle=shuffle)
        return generator

    def get_generator_fromnumpy_array(self, batch_size, data_array):

        class ThreadSafeIteratorFromArray:
            def __init__(self, data_array):
                self.lock = threading.Lock()
                self.data_array = data_array
                self.read_head = 0
                self.samples = data_array.shape[0]

            def next(self):
                with self.lock:
                    x = self.data_array[self.read_head:self.read_head+batch_size]
                    if x.shape[0] == batch_size:
                        self.read_head += batch_size
                        return x
                    else:
                        x = np.concatenate((x, self.data_array[0:batch_size-x.shape[0]]), axis=0)
                        self.read_head = batch_size-x.shape[0]
                        return x
        return ThreadSafeIteratorFromArray(data_array)

    def get_data_loader(self, dataset_name, shuffle):
        if dataset_name.upper() == 'CELEBA':
            img_rows = 64
            img_cols = 64
            channels = 3
            classes = None
            self.data_dir = '/is/ps2/pghosh/datasets/celebA64x64'
            self.train_dir = os.path.join(self.data_dir, 'train')
            self.validation_dir = os.path.join(self.data_dir, 'val')
            self.test_dir = os.path.join(self.data_dir, 'test')
            self.list_attrib_file = os.path.join(self.data_dir, 'list_attr_celeba.csv')
            img_size = (img_rows, img_cols)
            train_generator = self.get_generator_from_img_dir(self.train_dir, classes, img_size, shuffle)
            valid_generator = self.get_generator_from_img_dir(self.validation_dir, classes, img_size, shuffle)
            test_generator = self.get_generator_from_img_dir(self.test_dir, classes, img_size, shuffle)

        elif dataset_name.upper() == 'MNIST':
            self.data_dir = '/is/ps2/pghosh/datasets/mnist'
            dataset_path = os.path.join(self.data_dir, 'mnist_32x32.npz')
            img_rows = 32
            img_cols = 32
            channels = 1
            mnist_digits = np.load(dataset_path)
            train_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=mnist_digits['x_train'][:-10000])
            test_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=mnist_digits['x_test'])
            valid_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=mnist_digits['x_train'][10000:])
            self.labels = np.concatenate((mnist_digits['y_train'], mnist_digits['y_test']), axis=0)

        elif dataset_name.upper() == 'CIFAR_10' or dataset_name.upper() == 'CIFAR_100':
            self.data_dir = '/is/ps2/pghosh/datasets/cifar/'
            if dataset_name.upper() == 'CIFAR_10':
                dataset_path = '/is/ps2/pghosh/datasets/cifar/cifar_10.npz'
            else:
                dataset_path = '/is/ps2/pghosh/datasets/cifar/cifar_100.npz'
            img_rows = 32
            img_cols = 32
            channels = 3
            cifar_data = np.load(dataset_path)
            train_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=cifar_data['x_train'][:-10000])
            test_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=cifar_data['x_test'])
            valid_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size,
                                                   data_array=cifar_data['x_train'][10000:])
            self.labels = np.concatenate((cifar_data['y_train'][:, 0], cifar_data['y_test'][:, 0]), axis=0)
        elif dataset_name.upper() == 'SVHN':
            dataset_path = '/is/ps2/pghosh/datasets/SVHN'
            img_rows = 32
            img_cols = 32
            channels = 3
            x_train = loadmat(dataset_path + '/train_32x32.mat')
            x_train = np.transpose(x_train['X'], (3, 0, 1, 2)).astype('float32') / 255

            x_test = loadmat(dataset_path + '/test_32x32.mat')
            x_test = np.transpose(x_test['X'], (3, 0, 1, 2)).astype('float32') / 255
            train_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size, data_array=x_train)
            valid_generator = \
                self.get_generator_fromnumpy_array(batch_size=self.batch_size, data_array=x_test)
            test_generator = None
        else:
            raise ValueError("Invalid dataset name: " + str(dataset_name))

        imput_shape = (img_rows, img_cols, channels)
        if test_generator is not None:
            generators = (self.make_auto_encoder_gen_frm_keras_image_gen(train_generator),
                          self.make_auto_encoder_gen_frm_keras_image_gen(valid_generator),
                          self.make_auto_encoder_gen_frm_keras_image_gen(test_generator))
            steps_to_yield = ((train_generator.samples/self.batch_size), (valid_generator.samples/self.batch_size),
                              (test_generator.samples/self.batch_size))
            self.test_smpls = test_generator.samples
        else:
            generators = (self.make_auto_encoder_gen_frm_keras_image_gen(train_generator),
                          self.make_auto_encoder_gen_frm_keras_image_gen(valid_generator),
                          None)
            steps_to_yield = ((train_generator.samples / self.batch_size), (valid_generator.samples / self.batch_size),
                              None)

        self.train_smpls = train_generator.samples
        self.valid_smpls = valid_generator.samples
        return generators, imput_shape, steps_to_yield