import gzip
import json
import os
from typing import Iterable

import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer


def read_raw_dataset_games(file_path, successful_only=False):
    if file_path.endswith(".gz"):
        with gzip.open(file_path) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip("\n"))

                if not successful_only or (successful_only and game["status"] == "success"):
                    yield game
    else:
        with open(file_path) as f:
            for line in f:
                game = json.loads(line.strip("\n"))

                if not successful_only or (successful_only and game["status"] == "success"):
                    yield game


def get_split_id(dataset_path):
    return "train" if "train" in dataset_path else "test" if "test" in dataset_path else "valid"


@DatasetReader.register("dialogue_features")
class DialogueFeaturesDataset(DatasetReader):
    def __init__(self,
                 dialogue_features_path,
                 attribute_vectors_path,
                 dataset_type,
                 lazy=False,
                 make_vocab=False
                 ):
        super().__init__(lazy)
        self.attributes_indexer = {"attributes": SingleIdTokenIndexer(namespace="attributes", lowercase_tokens=True)}
        self.dialogue_features_path = dialogue_features_path
        self.attribute_vectors_path = attribute_vectors_path
        self.dataset_type = dataset_type
        self.make_vocab = make_vocab

        with open(os.path.join(self.attribute_vectors_path, "abstract_attribute_names.json")) as in_file:
            self.abstract_attribute_names = json.load(in_file)

        with open(os.path.join(self.attribute_vectors_path, "situated_attribute_names.json")) as in_file:
            self.situated_attribute_names = json.load(in_file)

        if self.dataset_type == "location":
            f_attributes = [
                i for i, a in enumerate(self.situated_attribute_names) \
                if a in ['bottom_image', 'right_image', 'left_image', 'top_image', 'center']
            ]
            self.attributes_filter = np.array(f_attributes)

    def get_dialogue_features(self, split_name, dialogue_id):
        dialogue_features_file = os.path.join(self.dialogue_features_path, split_name,
                                              "{}.npy".format(dialogue_id))

        return np.load(dialogue_features_file)

    def get_attribute_features(self, split_name, image_id, object_id):
        # attribute vectors are associated with a specific image
        attribute_vectors_file = os.path.join(self.attribute_vectors_path, split_name, image_id + ".npz")

        with np.load(attribute_vectors_file) as attribute_vectors:
            object_map = {obj_id: i for i, obj_id in enumerate(attribute_vectors.get("object2id"))}

            object_id = object_map.get(object_id)

            return attribute_vectors.get("abstract_attributes")[object_id], \
                   attribute_vectors.get("situated_attributes")[object_id]

    def attribute_ids2tokens(self, attribute_ids):
        if self.dataset_type == "abstract":
            return [self.abstract_attribute_names[i] for i in attribute_ids]
        else:
            tokens = []

            for i in attribute_ids:
                # if the current attribute id belongs to the abstract attributes
                if i in range(0, len(self.abstract_attribute_names)):
                    tokens.append(self.abstract_attribute_names[i])
                else:
                    # the current attribute id belongs to the situated attributes
                    real_id = i - len(self.abstract_attribute_names)
                    tokens.append(
                        self.situated_attribute_names[real_id]
                    )
        return tokens

    def _read(self, gw_dataset_path: str) -> Iterable[Instance]:
        # we need to create a dataset represented by situated attributes
        # so given the guesswhat dataset, we extract all the successful dialogues and we check if the dialogue
        # has corresponding dialogue states and attribute vectors
        # For each of these dialogues, we create a dataset in which the dialogue state is the input and the output
        # is the entire attribute vector for the "target" object
        split_name = get_split_id(gw_dataset_path)

        # We want to read only successful dialogues that are coherent with the final prediction
        for game in read_raw_dataset_games(gw_dataset_path, successful_only=True):
            game_id = str(game["id"])
            image_id = str(game["image"]["id"])
            dialogue_features_file = os.path.join(self.dialogue_features_path, split_name,
                                                  "{}.npy".format(game_id))
            target_object = next(filter(lambda o: o["id"] == game["object_id"], game["objects"]))

            # attribute vectors are associated with a specific image
            attribute_vectors_file = os.path.join(self.attribute_vectors_path, split_name, image_id + ".npz")

            with np.load(attribute_vectors_file) as attribute_vectors:
                object_map = {obj_id: i for i, obj_id in enumerate(attribute_vectors.get("object2id"))}

                dialogue_hidden_state = np.load(dialogue_features_file)

                object_id = object_map.get(target_object["id"])
                # every object should have at least the abstract attributes
                if self.dataset_type == "situated":
                    abstract = attribute_vectors["abstract_attributes"][object_id]
                    situated = attribute_vectors["situated_attributes"][object_id]
                    yield self.text_to_instance(
                        game_id,
                        game["qas"],
                        image_id,
                        target_object,
                        dialogue_hidden_state,
                        (abstract, situated)
                    )
                elif self.dataset_type == "situated_only":
                    situated = attribute_vectors["situated_attributes"][object_id]
                    yield self.text_to_instance(
                        game_id,
                        game["qas"],
                        image_id,
                        target_object,
                        dialogue_hidden_state,
                        (situated)
                    )
                elif self.dataset_type == "abstract":
                    abstract = attribute_vectors["abstract_attributes"][object_id]
                    yield self.text_to_instance(
                        game_id,
                        game["qas"],
                        image_id,
                        target_object,
                        dialogue_hidden_state,
                        (abstract)
                    )
                elif self.dataset_type == "location":
                    location = attribute_vectors["situated_attributes"][object_id][self.attributes_filter]

                    yield self.text_to_instance(
                        game_id,
                        game["qas"],
                        image_id,
                        target_object,
                        dialogue_hidden_state,
                        (location)
                    )

    def text_to_instance(self, game_id, qas, image_id, target_object, dialogue_features,
                         target_attributes=None) -> Instance:
        metadata = {
            "game_id": game_id,
            "image_id": image_id,
            "target_object": target_object,
            "qas": qas
        }

        instance = {
            "metadata": MetadataField(metadata),
            "dialogue_states": ArrayField(dialogue_features)
        }

        if target_attributes is not None:
            instance["target_attributes"] = ArrayField(np.concatenate(target_attributes)) if isinstance(
                target_attributes, tuple) else \
                ArrayField(target_attributes)

        return Instance(instance)
