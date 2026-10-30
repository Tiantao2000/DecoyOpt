"""Metric regression checks against scikit-learn indicator calculations."""
import unittest

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from decoyopt.metrics import choose_mcc_threshold, multilabel_metrics


class MetricsTest(unittest.TestCase):
    def test_multilabel_matches_indicator_metrics(self):
        truth = [{"1.1.1.1", "2.2.2.2"}, {"1.1.1.1"}, {"non_enzyme"}, {"3.3.3.3"}]
        prediction = [{"1.1.1.1"}, {"non_enzyme"}, {"2.2.2.2"}, set()]
        encoder = MultiLabelBinarizer().fit(truth + prediction)
        a, b = encoder.transform(truth), encoder.transform(prediction)
        result = multilabel_metrics(truth, prediction, 4, True)
        expected = {
            "weighted_precision": precision_score(a, b, average="weighted", zero_division=0),
            "weighted_recall": recall_score(a, b, average="weighted", zero_division=0),
            "weighted_f1": f1_score(a, b, average="weighted", zero_division=0),
            "macro_f1": f1_score(a, b, average="macro", zero_division=0),
            "overall_mcc": matthews_corrcoef(a.ravel(), b.ravel()),
        }
        for metric, value in expected.items():
            self.assertAlmostEqual(result[metric], value)

    def test_rejected_enzymes_remain_false_negatives(self):
        result = multilabel_metrics([{"1.1.1.1"}, {"2.2.2.2"}], [set(), {"2.2.2.2"}], 4)
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["n_false_negatives"], 1)
        self.assertEqual(result["weighted_recall"], 0.5)

    def test_threshold_handles_ties_as_groups(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        scores = np.array([0., 1., 1., 2., 3., 3.])
        threshold = choose_mcc_threshold(y, scores)
        candidates = sorted(set(scores), reverse=True)
        expected = max(candidates, key=lambda t: matthews_corrcoef(y, scores >= t))
        self.assertEqual(threshold, expected)


if __name__ == "__main__":
    unittest.main()
