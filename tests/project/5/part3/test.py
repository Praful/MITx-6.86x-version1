import unittest
import numpy as np
import agent_tabular_ql as agent


# Instructions:
#   Put all files in the same folder as agent_tabular_ql.py and other files.
#   To run: python <this file>
#
# Description:
#   The tests are for testing agent_tabular_ql functions tabular_q_learning() and
#   epsilon_greedy().


class TestAgentTabularQL(unittest.TestCase):
    #  Uncomment to skip test; add line to other tests if you want to skip them
    #  #  unittest.skip('Skip this test')
    def setUp(self):
        self.q_func = np.load('./q_funct.npy')


    #  @unittest.skip('Skip this test')
    def test_tabular_q_learning1(self):
        current_state_1=15
        current_state_2=0
        action_index=0
        object_index=0
        next_state_1=15
        next_state_2=0
        reward=-0.11

        expected_q = -0.0209

        agent.tabular_q_learning(self.q_func, current_state_1, current_state_2,
            action_index, object_index, reward, next_state_1, next_state_2, False)

        actual_q = self.q_func[current_state_1, current_state_2, action_index, object_index]

        self.assertAlmostEqual(actual_q, expected_q, 8,
                         f'q value: got {actual_q}, expected {expected_q}')

    def test_tabular_q_learning2(self):
        current_state_1=13
        current_state_2=1
        action_index=0
        object_index=1
        next_state_1=15
        next_state_2=0
        reward=-0.11

        expected_q = -0.011

        agent.tabular_q_learning(self.q_func, current_state_1, current_state_2,
            action_index, object_index, reward, next_state_1, next_state_2, False)

        actual_q = self.q_func[current_state_1, current_state_2, action_index, object_index]

        self.assertAlmostEqual(actual_q, expected_q, 8,
                         f'q value: got {actual_q}, expected {expected_q}')


    def test_epsilon_greedy1(self):
        state_1=15
        state_2=0

        expected_action_index = 0
        expected_object_index = 1
        actual_action_index, actual_object_index = agent.epsilon_greedy(state_1, state_2, self.q_func, 0)
        self.assertEqual(expected_action_index, actual_action_index, f'action index: got {actual_action_index}, expected {expected_action_index}')
        self.assertEqual(expected_object_index, actual_object_index, f'object index: got {actual_object_index}, expected {expected_object_index}')

    def test_epsilon_greedy2(self):
        state_1=0
        state_2=0
        #  print(self.q_func[0, 0])

        expected_action_index = 2
        expected_object_index = 1
        actual_action_index, actual_object_index = agent.epsilon_greedy(state_1, state_2, self.q_func, 0)
        self.assertEqual(expected_action_index, actual_action_index, f'action index: got {actual_action_index}, expected {expected_action_index}')
        self.assertEqual(expected_object_index, actual_object_index, f'object index: got {actual_object_index}, expected {expected_object_index}')

if __name__ == '__main__':
    unittest.main()
