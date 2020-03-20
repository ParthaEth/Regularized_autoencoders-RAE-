from my_utility import save_restore_model_state
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

import unittest


class TestModelSaveRestore(unittest.TestCase):

    def test_keras_model_save_restore(self):
        current_epoch = 3

        def get_model():
            I = Input(shape=(200,))
            x = Dense(10)(I)
            x = Dense(13)(x)
            x = Dense(100)(x)

            model = Model(inputs=I, outputs=x)
            model.compile(optimizer=Adam(lr=1e3), loss='mse')
            return model

        model = get_model()
        model.fit(x=np.random.uniform(-1, 1, size=(32, 200)), y=np.random.uniform(-1, 1, size=(32, 100)), epochs=100)
        save_restore_model_state.save_model_state(model=model, checkpoint_path='./', epoch=current_epoch)

        another_model = get_model()

        current_model_weights = model.get_weights()
        for i in range(len(current_model_weights)):
            self.assertTrue((np.abs(current_model_weights[i]-another_model.get_weights()[i]) > 0.1).all())

        restored_epoch = save_restore_model_state.restore_model_state(another_model, checkpoint_path='./')
        for i in range(len(current_model_weights)):
            np.testing.assert_almost_equal(current_model_weights[i], another_model.get_weights()[i], decimal=7)
        self.assertEquals(restored_epoch, current_epoch)


if __name__ == '__main__':
    unittest.main()
