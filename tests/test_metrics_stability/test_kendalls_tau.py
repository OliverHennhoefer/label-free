import unittest

from labelfree.metrics.stability.correlation import kendalls_tau


class TestSuiteKentallsTau(unittest.TestCase):

    def test_separate_lists_equal_rank(self):

        scores1 = [1, 4, 5, 2, 6, 9, 3, 7, 8]
        scores2 = [1, 4, 5, 2, 6, 9, 3, 7, 8]

        tau = kendalls_tau(scores1, scores2)
        self.assertEqual(tau, 1)

    def test_separate_lists_one_rank_disagreement(self):
        scores1 = [1, 4, 5, 2, 9, 6, 3, 7, 8]
        scores2 = [1, 4, 5, 2, 6, 9, 3, 7, 8]

        tau = kendalls_tau(scores1, scores2)
        self.assertEqual(tau, 0.7222222222222222)

    def test_nested_list_equal_rank(self):

        scores1 = [1, 4, 5, 2, 6, 9, 3, 7, 8]
        scores2 = [1, 4, 5, 2, 6, 9, 3, 7, 8]

        tau = kendalls_tau([scores1, scores2])
        self.assertEqual(tau, 1)

    def test_nested_list_one_rank_disagreement(self):
        scores1 = [1, 4, 5, 2, 9, 6, 3, 7, 8]
        scores2 = [1, 4, 5, 2, 6, 9, 3, 7, 8]

        tau = kendalls_tau([scores1, scores2])
        self.assertEqual(tau, 0.7222222222222222)

    def test_nested_list_float_scores(self):
        scores1 = [0.1, 0.4, 0.5, 0.2, 0.9, 0.6, 0.3, 0.7, 0.8]
        scores2 = [0.1, 0.4, 0.5, 0.2, 0.6, 0.9, 0.3, 0.7, 0.8]

        tau = kendalls_tau([scores1, scores2])
        self.assertEqual(tau, 0.7222222222222222)


if __name__ == "__main__":
    unittest.main()
