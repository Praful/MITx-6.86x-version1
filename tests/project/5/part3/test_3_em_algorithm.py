import numpy as np
import common
import unittest
import naive_em
import estep_test_input_0
import estep_test_input_5
import mstep_test_input_0
import mstep_test_input_3

# Instructions:
#   Put all files in the same folder as naive_em.py and other files.
#   To run: python test_3_em_algorithm.py
#
# Description:
#   The tests use input and output from submissions. These are used to check your 
#   results.
#
class TestEMAlgorithm(unittest.TestCase):
    #  Uncomment to skip test; add line to other tests if you want to skip them
    #  @unittest.skip('Skip this test')
    def test_estep_toy_dataset(self):
        X = np.loadtxt("toy_data.txt")
        mixture, post = common.init(X, 3, 0)
        post, ll = naive_em.estep(X, mixture)
        self.assertEqual(np.isclose(ll, -1388.0818), True)


    def run_estep_test_input(self, test):
        X, K, mixture, expected_post, expected_ll = test.data()
        post, ll = naive_em.estep(X, mixture)

        self.assertEqual(np.isclose(ll, expected_ll), True)
        self.assertEqual(np.allclose(post, expected_post), True)

    def run_mstep_test_input(self, test):
        X, expected_mixture, post = test.data()
        mixture = naive_em.mstep(X, post)

        self.assertEqual(np.allclose(mixture.mu, expected_mixture.mu), True)
        self.assertEqual(np.allclose(mixture.var, expected_mixture.var), True)
        self.assertEqual(np.allclose(mixture.p, expected_mixture.p), True)

    def test_estep_input_0(self):
        self.run_estep_test_input(estep_test_input_0)

    def test_estep_input_5(self):
        self.run_estep_test_input(estep_test_input_5)

    def test_mstep_input_0(self):
        self.run_mstep_test_input(mstep_test_input_0)

    def test_mstep_input_3(self):
        self.run_mstep_test_input(mstep_test_input_3)


if __name__ == '__main__':
    unittest.main()
