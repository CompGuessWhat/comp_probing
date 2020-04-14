from typing import Dict, Optional, List

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator, InitializerApplicator

from comp_probing.metrics.multi_label import MultiLabelF1Measure


# Mock module used for generating prediction
@Model.register("random_probe")
class RandomProbe(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 output_dim: int,
                 thresholds: List[float] = (0.5, 0.75, 0.9),
                 average_metrics=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab, regularizer)
        initializer(self)
        self.output_dim = output_dim
        self.f1_metrics = {
            t: MultiLabelF1Measure(t, average_metrics) for t in thresholds
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        for metric_thresh, metric_obj in self.f1_metrics.items():
            f1, precision, recall = metric_obj.get_metric(reset)

            metrics["f1_{}".format(metric_thresh)] = f1
            metrics["precision_{}".format(metric_thresh)] = precision
            metrics["recall_{}".format(metric_thresh)] = recall

        return metrics

    def forward(self, metadata, dialogue_states, target_attributes=None) -> Dict[str, torch.Tensor]:
        batch_size = dialogue_states.shape[0]

        # samples from a Uniform distribution
        pred_attributes_logits = torch.empty(batch_size, self.output_dim).uniform_(0, 1)
        output_dict = {
            "pred_attributes": pred_attributes_logits
        }

        if target_attributes is not None:
            for metric in self.f1_metrics.values():
                metric(output_dict["pred_attributes"], target_attributes)

        return output_dict


@Model.register("linear_probe")
class LinearProbe(Model):
    def __init__(self, hidden_state_dim: int, output_dim: int, vocab: Vocabulary,
                 thresholds: List[float] = (0.5, 0.75, 0.9),
                 average_metrics=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab, regularizer)
        self.probe = torch.nn.Linear(hidden_state_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        initializer(self)

        self.f1_metrics = {
            t: MultiLabelF1Measure(t, average_metrics) for t in thresholds
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        for metric_thresh, metric_obj in self.f1_metrics.items():
            f1, precision, recall = metric_obj.get_metric(reset)

            metrics["f1_{}".format(metric_thresh)] = f1
            metrics["precision_{}".format(metric_thresh)] = precision
            metrics["recall_{}".format(metric_thresh)] = recall

        return metrics

    def forward(self, metadata, dialogue_states, target_attributes=None) -> Dict[str, torch.Tensor]:
        if dialogue_states.dim() == 3 and dialogue_states.size(1) == 1:
            dialogue_states = dialogue_states.squeeze(1)
        # apply the probe
        pred_attributes_logits = self.probe(dialogue_states)
        output_dict = {
            "pred_attributes": self.sigmoid(pred_attributes_logits)
        }

        if target_attributes is not None:
            loss = F.binary_cross_entropy_with_logits(pred_attributes_logits, target_attributes)

            output_dict["loss"] = loss

            for metric in self.f1_metrics.values():
                metric(output_dict["pred_attributes"], target_attributes)

        return output_dict
