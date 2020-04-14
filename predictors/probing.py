from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor

from comp_probing.datasets.guesswhat import DialogueFeaturesDataset


@Predictor.register("probing_attribute")
class ProbingAttributePredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        dialogue_id = json_dict["dialogue_id"]
        image_id = json_dict["image_id"]
        qas = json_dict["qas"]
        object_id = json_dict["target_object"]["id"]
        split_name = json_dict.get("split_name", "valid")

        dialogue_hidden_state = self._dataset_reader.get_dialogue_features(split_name, dialogue_id)
        abstract, situated = self._dataset_reader.get_attribute_features(split_name, image_id, object_id)

        return self._dataset_reader.text_to_instance(
            dialogue_id,
            qas,
            image_id,
            json_dict["target_object"],
            dialogue_hidden_state,
            (abstract,) if self._dataset_reader.dataset_type == "abstract" else (abstract, situated)
        )

    def __init__(self, model: Model, dataset_reader: DialogueFeaturesDataset):
        super().__init__(model, dataset_reader)
