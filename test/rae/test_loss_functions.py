import unittest
import numpy as np
from models import rae
import keras.backend as K


class TestLossfunctions(unittest.TestCase):

    def test_embedding_loss(self):
        z = np.ones(shape=(10, 8, 8, 1))

        emb_per_pix_loss = K.eval(rae.embeddig_loss(z)())
        np.testing.assert_almost_equal(emb_per_pix_loss, 1.0, decimal=7)

        z = np.random.uniform(-1, 1, size=(10, 8, 8, 1))
        expected_embd_loss = np.mean(np.square(z))
        emb_per_pix_loss = K.eval(rae.embeddig_loss(z)())
        np.testing.assert_almost_equal(expected_embd_loss, emb_per_pix_loss)

    def test_per_pix_recon_loss(self):
        y_true = np.ones(shape=(10, 64, 64, 3))
        y_pred = np.zeros(shape=(10, 64, 64, 3))
        computed_recon_per_pix_loss = K.eval(rae.per_pix_recon_loss(y_true, y_pred))

        np.testing.assert_almost_equal(3.0, computed_recon_per_pix_loss)

        y_true = np.random.uniform(-1, 1, size=(10, 64, 64, 3))
        y_pred = np.random.uniform(-1, 1, size=(10, 64, 64, 3))
        true_recon_per_pix_loss = np.mean(np.sum(np.square(y_true - y_pred), axis=-1))

        computed_recon_per_pix_loss = K.eval(rae.per_pix_recon_loss(y_true, y_pred))
        np.testing.assert_almost_equal(true_recon_per_pix_loss, computed_recon_per_pix_loss)

    def test_total_loss(self):
        z = np.ones(shape=(10, 8, 8, 1))
        y_true = np.ones(shape=(10, 64, 64, 3))
        y_pred = np.zeros(shape=(10, 64, 64, 3))
        total_loss_expected = 1 + 3
        computed_total_loss = K.eval(rae.total_loss(z, 1.0)(y_true, y_pred))
        np.testing.assert_almost_equal(total_loss_expected, computed_total_loss)


if __name__ == '__main__':
    unittest.main()