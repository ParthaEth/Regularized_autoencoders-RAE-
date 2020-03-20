from my_utility import estimate_density_and_sample
import numpy as np

import unittest


class TestDensityEstimator(unittest.TestCase):

    def test_density_estimator_and_sampling(self):
        scale = 2.0
        data_dim = int(3)
        train_data = np.random.multivariate_normal(mean=np.zeros(data_dim), cov=scale*np.identity(data_dim),
                                                   size=int(1e4))
        sampler = estimate_density_and_sample.DensityEstimator(training_set=train_data, method_name='GMM_Dirichlet')
        smpls = sampler.get_samples(n_samples=1000)
        estimated_covar = np.cov(smpls.T)
        np.testing.assert_almost_equal(scale*np.identity(data_dim), estimated_covar, 1)


if __name__ == '__main__':
    unittest.main()
