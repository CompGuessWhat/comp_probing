from typing import Optional

import numpy as np
import torch
from allennlp.training.metrics import Metric
from overrides import overrides


# Measures implemented following the ones described in https://pdfs.semanticscholar.org/6b56/91db1e3a79af5e3c136d2dd322016a687a0b.pdf
class MultiLabelAccuracy(Metric):
    def __init__(self, threshold=0.5):
        self.accuracies = []
        self.threshold = threshold

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions_t = (predictions > self.threshold).float()
        correct = predictions_t.eq(gold_labels).float() * gold_labels

        total_count = (predictions_t == 1).float().sum(-1) + (gold_labels == 1).float().sum(-1) - correct.sum(-1)

        self.accuracies.extend(correct.sum(-1) / total_count)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.accuracies:
            accuracy = np.average(self.accuracies)
        else:
            accuracy = 0.0

        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.accuracies.clear()


class MultiLabelF1Measure(Metric):
    def __init__(self, threshold=0.5, average=True):
        self.recall = []
        self.precision = []
        self.threshold = threshold
        self.average = average

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions_t = (predictions > self.threshold).float()
        # y_i_z_i: total number of correct predictions
        y_i_zi = (predictions_t.eq(gold_labels).float() * gold_labels).sum(-1)
        # z_i: total number of actual labels
        z_i = gold_labels.sum(-1)
        # y_i: total number of predictions
        y_i = predictions_t.sum(-1)

        self.precision.extend(
            torch.where(z_i != 0, y_i_zi / z_i, torch.zeros_like(z_i))
        )

        self.recall.extend(
            torch.where(y_i != 0, y_i_zi / y_i, torch.zeros_like(y_i))
        )

    def _compute_f1(self, precision, recall):
        precision = np.array(precision)
        recall = np.array(recall)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            np.zeros_like(precision)
        )

        return f1_scores

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.recall and self.precision:
            f1_score = self._compute_f1(self.precision, self.recall)
            if self.average:
                f1_score = float(np.average(f1_score))
                recall = float(np.average(self.recall))
                precision = float(np.average(self.precision))
            else:
                precision = np.array(self.precision)
                recall = np.array(self.recall)

        else:
            f1_score, precision, recall = 0.0, 0.0, 0.0
        if reset:
            self.reset()
        return f1_score, precision, recall

    @overrides
    def reset(self):
        self.precision.clear()
        self.recall.clear()


def test_accuracy(predictions, targets):
    metric_05 = MultiLabelAccuracy(0.5)
    metric_07 = MultiLabelAccuracy(0.7)

    metric_05(predictions, targets)

    print(metric_05.get_metric())

    metric_07(predictions, targets)

    print(metric_07.get_metric())


def test_f1(predictions, targets):
    metric_05 = MultiLabelF1Measure(0.5, average=False)
    metric_07 = MultiLabelF1Measure(0.7, average=False)

    metric_05(predictions, targets)

    print(metric_05.get_metric())

    metric_07(predictions, targets)

    print(metric_07.get_metric())


if __name__ == "__main__":
    predictions = torch.tensor([
        [0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])

    targets = torch.tensor([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0]
    ]).float()

    print("Testing MultiLabel Accuracy scores")
    test_accuracy(predictions, targets)

    print("Testing F1 scores")
    test_f1(predictions, targets)
