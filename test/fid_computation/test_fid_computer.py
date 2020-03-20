import unittest
import numpy as np
from my_utility import my_callbacks
from dataloaders.dataloader import DataLoader
import keras.backend as K


class TestFidComputation(unittest.TestCase):

    def test_truesamples_celeba(self):
        batch_size = 100
        (train_generator, validation_generator, _), input_shape, (train_steps, validation_steps, _) = \
            DataLoader(batch_size=batch_size).get_data_loader(dataset_name='celeba')

        num_evaluation_batches = 192
        for i in range(num_evaluation_batches):
            if i == 0:
                original_imgs = train_generator.next()[0]
                predicted_iamges = validation_generator.next()[0]
            else:
                original_imgs = np.concatenate((original_imgs, train_generator.next()[0]), axis=0)
                predicted_iamges = np.concatenate((predicted_iamges, validation_generator.next()[0]), axis=0)

        fid_computer=my_callbacks.FIDComputer(max_smpls_in_batch=32)
        natural_smpl_fid = fid_computer.get_fid(K.get_session(), original_imgs, predicted_iamges)
        fid_thress = 3
        print natural_smpl_fid
        self.assertTrue(natural_smpl_fid < fid_thress, 'Natural sample fid must be very low but when computed on ' +
                        str(num_evaluation_batches*batch_size) + " samples, it is larger than threshold provided. "
                                                                 "Computed FID is " + str(natural_smpl_fid) + " While "
                                                                 "threshold is " + str(fid_thress))

    def test_fid_known_samples(self):
        fid_expected = 7.7
        true_images = np.load('ft/1.npy')
        generated_iamges = np.load('ft/2.npy')
        fid_computer = my_callbacks.FIDComputer(max_smpls_in_batch=300)
        smpl_fid = fid_computer.get_fid(K.get_session(), true_images, generated_iamges)
        print smpl_fid
        self.assertTrue(smpl_fid < fid_expected, 'Computed FID is too high')


if __name__ == '__main__':
    unittest.main()