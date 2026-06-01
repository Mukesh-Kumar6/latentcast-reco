import unittest

from evaluation.metrics import ndcg_at_k, recall_at_k


class RankingMetricsTest(unittest.TestCase):
    def test_recall_at_k_counts_relevant_items(self):
        self.assertEqual(
            recall_at_k(["a", "b", "c"], {"a", "c", "d", "e"}, k=3),
            0.5,
        )

    def test_ndcg_at_k_is_one_for_ideal_binary_ranking(self):
        self.assertEqual(ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3), 1.0)

    def test_ndcg_at_k_penalizes_late_relevant_items(self):
        ideal = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
        delayed = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
        self.assertLess(delayed, ideal)

    def test_duplicate_recommendations_do_not_score_twice(self):
        self.assertEqual(recall_at_k(["a", "a", "b"], {"a", "b"}, k=2), 1.0)


if __name__ == "__main__":
    unittest.main()
