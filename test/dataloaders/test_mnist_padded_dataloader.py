from dataloaders.dataloader import DataLoader
import unittest
import keras.datasets.mnist
import numpy as np
import matplotlib.pyplot as plt

class TestPaddedMnist(unittest.TestCase):
    def test_mnist_padding_and_scaling(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        (train_loader, test_loader), _, (train_steps, test_steps) = DataLoader(batch_size=1000).get_data_loader('mnist')
        for i in range(train_steps):
            np.testing.assert_almost_equal(x_train[i*1000:(1+i)*1000],
                                           train_loader.next()[0][:, 2:-2, 2:-2, 0]*255.0, 5)

        for i in range(test_steps):
            np.testing.assert_almost_equal(x_test[i*1000:(1+i)*1000], test_loader.next()[0][:, 2:-2, 2:-2, 0]*255.0, 5)



if __name__ == '__main__':
    unittest.main()