import numpy as np
import common
import unittest
import em
import test_solutions as ts
import estep_test_input_0
import estep_test_input_5
import mstep_test_input_0
import mstep_test_input_3
import bic_test_input_fixed
import bic_test_input_random
import estep_2_test_input_3
import mstep_2_test_input_0
import mstep_2_test_input_2
import mstep_2_test_input_3


# Instructions:
#   Put all files in the same folder as em.py and other files.
#   To run: python <this file>
#
# Description:
#   The tests use the provided test_solutions.txt turned into a Python file and some 
#   output from submissions.


class TestEMMatrixAlgorithm(unittest.TestCase):
    #  Uncomment to skip test; add line to other tests if you want to skip them
    #  #  unittest.skip('Skip this test')
    def setUp(self):
        self.X = np.loadtxt("toy_data.txt")

    #  @unittest.skip('Skip this test')
    def test_estep_toy_dataset(self):
        mixture, post = common.init(self.X, 3, 0)
        post, ll = em.estep(self.X, mixture)
        expected_ll = -1388.0818
        self.assertEqual(np.isclose(ll, expected_ll),
                         True, f'Log likelihood: got {ll}, expected {expected_ll}')

    #  @unittest.skip('this is from the test solution')
    def test_estep_test_solution(self):
        post, ll = em.estep(ts.X, ts.mixture_initial)

        self.assertEqual(np.allclose(post, ts.post_first_estep),
                         True, f'Post not as expected:\nGot{post}\nExpected {ts.post_first_estep}')
        self.assertAlmostEqual(ll, ts.ll_first_estep, 8,
                         f'Log likelihood: got {ll}, expected {ts.ll_first_estep}')

    #  @unittest.skip('this is from the test solution')
    def test_mstep_test_solution(self):
        expected_mixture = ts.mixture_first_mstep
        mixture = em.mstep(ts.X, ts.post_first_estep, ts.mixture_initial)

        with self.subTest(msg='Check mu'):
            self.assertEqual(np.allclose(mixture.mu, expected_mixture.mu), True, f'mu not as expected.\nExpected: {expected_mixture.mu}\nGot:{mixture.mu}')

        with self.subTest(msg='Check var'):
            self.assertEqual(np.allclose(mixture.var, expected_mixture.var), True, f'var not as expected.\nExpected: {expected_mixture.var}\nGot:{mixture.var}')

        with self.subTest(msg='Check p'):
            self.assertEqual(np.allclose(mixture.p, expected_mixture.p), True, f'p not as expected.\nExpected: {expected_mixture.p}\nGot:{mixture.p}')


    # Run a test whose data is a Python file, which was created from  a submission
    def run_estep_test_input(self, test):
        X, K, mixture, expected_post, expected_ll = test.data()
        post, ll = em.estep(X, mixture)

        self.assertEqual(np.allclose(post, expected_post),
                         True, f'Post not as expected:\nGot {post}\nExpected {expected_post}')
        self.assertEqual(np.isclose(ll, expected_ll),
                         True, f'Log likelihood: got {ll}, expected {expected_ll}')

    # Run a test whose data is a Python file, which was created from  a submission
    def run_mstep_test_input(self, test):
        X, K, expected_mixture, post = test.data()
        mixture = em.mstep(X, post, expected_mixture)

        with self.subTest(msg='Check mu'):
            self.assertEqual(np.allclose(mixture.mu, expected_mixture.mu), True, f'mu not as expected.\nExpected: {expected_mixture.mu}\nGot:{mixture.mu}')

        with self.subTest(msg='Check var'):
            self.assertEqual(np.allclose(mixture.var, expected_mixture.var), True, f'var not as expected.\nExpected: {expected_mixture.var}\nGot:{mixture.var}')

        with self.subTest(msg='Check p'):
            self.assertEqual(np.allclose(mixture.p, expected_mixture.p), True, f'p not as expected.\nExpected: {expected_mixture.p}\nGot:{mixture.p}')

    #  @unittest.skip('Skip this test')
    def test_estep_2_input_3(self):
        self.run_estep_test_input(estep_2_test_input_3)

    #  @unittest.skip('Skip this test')
    def test_estep_input_0(self):
        self.run_estep_test_input(estep_test_input_0)

    #  @unittest.skip('Skip this test')
    def test_estep_input_5(self):
        self.run_estep_test_input(estep_test_input_5)

    #  @unittest.skip('Skip this test')
    def test_mstep_input_0(self):
        self.run_mstep_test_input(mstep_2_test_input_0)

    #  @unittest.skip('Skip this test')
    def test_mstep_input_2(self):
        self.run_mstep_test_input(mstep_2_test_input_2)

    #  @unittest.skip('Skip this test')
    def test_mstep_input_3(self):
        self.run_mstep_test_input(mstep_2_test_input_3)

    def test_run_test_solution(self):
        X, mixture, post = ts.X, ts.mixture_first_run, ts.post_first_run
        expected_cost = ts.ll_first_run

        new_mixture, cost = em.run(X, mixture, post)

        self.assertEqual(np.isclose(cost, expected_cost),
        True, f'Cost: got {cost}, expected {expected_cost}')



if __name__ == '__main__':
    unittest.main()
