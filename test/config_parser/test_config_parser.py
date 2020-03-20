from my_utility import config_parser
import unittest


class TestConfigParsing(unittest.TestCase):
    def test_config_parser(self):
        configurations = \
            {0: [{'base_model_name': "rae"},
                                       {'expt_name': 'l2_regularization'},

                                       {'expt_name': 'l2_regularization'}
                   ],

             1: [{'base_model_name': "rae"},
                                       {'expt_name': 'spectral_normalization'}
                   ],

             2: [{'base_model_name': "rae"},
                                       {'expt_name': 'l2_regularization'},

                                       {'expt_name': 'l2_regularization'}
                   ],
            }

        mj_i, min_i = config_parser.get_config_idxs(0, configurations)
        self.assertEquals((mj_i, min_i), (0, 1))
        pid = config_parser.get_process_id_given_mj_minor_idxs(mj_i, min_i, configurations)
        self.assertEquals(pid, 0)

        mj_i, min_i = config_parser.get_config_idxs(1, configurations)
        self.assertEquals((mj_i, min_i), (0, 2))
        pid = config_parser.get_process_id_given_mj_minor_idxs(mj_i, min_i, configurations)
        self.assertEquals(pid, 1)

        mj_i, min_i = config_parser.get_config_idxs(2, configurations)
        self.assertEquals((mj_i, min_i), (1, 1))
        pid = config_parser.get_process_id_given_mj_minor_idxs(mj_i, min_i, configurations)
        self.assertEquals(pid, 2)

        mj_i, min_i = config_parser.get_config_idxs(3, configurations)
        self.assertEquals((mj_i, min_i), (2, 1))
        pid = config_parser.get_process_id_given_mj_minor_idxs(mj_i, min_i, configurations)
        self.assertEquals(pid, 3)

        mj_i, min_i = config_parser.get_config_idxs(4, configurations)
        self.assertEquals((mj_i, min_i), (2, 2))
        pid = config_parser.get_process_id_given_mj_minor_idxs(mj_i, min_i, configurations)
        self.assertEquals(pid, 4)

if __name__ == '__main__':
    unittest.main()