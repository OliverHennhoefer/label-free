import unittest

from labelfree.metrics.stability.correlation import spearmans_rho


class TestSuiteSpearmansRho(unittest.TestCase):

    def test_separate_lists_equal_rank(self):

        scores1 = [0.5, 0.6, 0.88, 0.1, 0.99, 1.999, 0.1453]
        scores2 = [0.51, 0.61, 0.89, 0.11, 0.9, 1.965, 0.1234]

        tau = spearmans_rho(scores1, scores2)
        self.assertEqual(tau, 1)

    def test_separate_lists_one_rank_disagreement(self):

        scores1 = [0.5, 0.6, 0.88, 0.1, 0.99, 0.999, 0.1453]
        scores2 = [0.51, 0.61, 0.19, 0.11, 0.9, 0.965, 0.1234]

        tau = spearmans_rho(scores1, scores2)
        self.assertEqual(tau, 0.8928571428571429)

    def test_nested_list_equal_rank(self):

        scores1 = [0.5, 0.6, 0.88, 0.1, 0.99, 1.999, 0.1453]
        scores2 = [0.51, 0.61, 0.89, 0.11, 0.9, 1.965, 0.1234]

        tau = spearmans_rho([scores1, scores2])
        self.assertEqual(tau, 1)

    def test_nested_list_one_rank_disagreement(self):

        scores1 = [0.5, 0.6, 0.88, 0.1, 0.99, 0.999, 0.1453]
        scores2 = [0.51, 0.61, 0.19, 0.11, 0.9, 0.965, 0.1234]

        tau = spearmans_rho([scores1, scores2])
        self.assertEqual(tau, 0.8928571428571429)


if __name__ == "__main__":
    unittest.main()
