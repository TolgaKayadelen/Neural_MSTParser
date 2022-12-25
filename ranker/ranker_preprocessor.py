"""Preprocessor module that creates tf.data.Dataset for the ranker"""

import os
import itertools
import tensorflow as tf
import numpy as np

from input.embeddor import Embeddings
from input import preprocessor
from util import reader, writer
from util.nn import nn_utils
from typing import List, Dict
from tagset.morphology import morph_tags

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Dataset = tf.data.Dataset
Example = tf.train.Example

# _DATA_DIR = "./data/UDv29/test/tr"
_DATA_DIR = "./ranker"

_CASE_VALUES = {"abl": 0, "acc": 1, "dat": 2, "equ": 3, "gen": 4, "ins": 5, "loc": 6, "nom": 7}
_PERSON_VALUES = {"1": 0, "2": 1, "3": 2}
_VOICE_VALUES = {"cau": 0, "pass": 1, "rcp": 2, "rfl": 3}
_VERBFORM_VALUES = {"conv": 0, "part": 1, "vnoun": 2}

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self,
               name: str,
               values: List[int]=[],
               dtype=tf.int64):
    # the name of this feature.
    self.name = name
    # the value indices of this feature. This is generated on the fly for each data point in the
    # data during training/test based on a mapping.
    self.values = values
    # feature's data type.
    self.dtype = dtype

class RankerPreprocessor(preprocessor.Preprocessor):
  """Prepares training data for ranker as tf.examples"""

  def __init__(self, *, word_embeddings: Embeddings = None,
               features: List[str] = ["words", "pos", "morph", "dep_labels"],
               data_path: str):
    super(RankerPreprocessor, self).__init__(word_embeddings=word_embeddings,
                                             features=features,
                                             labels=[])
    self.data_path = data_path
    self.feature_mappings = self.get_index_mappings_for_features(self.features)

  @property
  def train_data(self):
    return reader.ReadRankerTextProto(os.path.join(_DATA_DIR, self.data_path))



  def make_tf_example(self, datapoint) -> tf.train.Example:
    example = tf.train.Example()
    features = example.features
    next_token_ids = self.numericalize(values=[t.word for t in datapoint.features.next_token],
                                       mapping=self.feature_mappings["words"])
    prev_token_ids = self.numericalize(values=[t.word for t in datapoint.features.previous_token],
                                       mapping=self.feature_mappings["words"])
    next_token_pos_ids = self.numericalize(values=[t.pos for t in datapoint.features.next_token],
                                           mapping=self.feature_mappings["pos"])
    prev_token_pos_ids = self.numericalize(values=[t.pos for t in datapoint.features.previous_token],
                                           mapping=self.feature_mappings["pos"])

    features.feature["word_id"].int64_list.CopyFrom(self._int_feature([datapoint.word_id]))
    features.feature["pos_id"].int64_list.CopyFrom(self._int_feature(
      self.numericalize(values=[datapoint.features.pos], mapping=self.feature_mappings["pos"])))
    features.feature["next_token_ids"].int64_list.CopyFrom(self._int_feature(next_token_ids))
    features.feature["prev_token_ids"].int64_list.CopyFrom(self._int_feature(prev_token_ids))
    features.feature["next_token_pos_ids"].int64_list.CopyFrom(self._int_feature(next_token_pos_ids))
    features.feature["prev_token_pos_ids"].int64_list.CopyFrom(self._int_feature(prev_token_pos_ids))
    features.feature["hypotheses"].int64_list.CopyFrom(self._int_feature([h.label_id for h in datapoint.hypotheses]))

    if datapoint.features.morphology:
      for feat in ["case", "voice", "person", "verbform"]:
        features.feature[feat].float_list.CopyFrom(
          self._float_feature(self._morph_feature(datapoint.features.morphology, feat)))

    features.feature["word_strings"].bytes_list.CopyFrom(self._bytes_feature([datapoint.word.encode("utf-8")]))
    features.feature["hypotheses_strings"].bytes_list.CopyFrom(
      self._bytes_feature([h.label.encode("utf-8") for h in datapoint.hypotheses]))

    # next_token_vectors, prev_token_vectors = [], []
    # for token in  datapoint.features.next_token:
      # next_token_vectors.extend(self._morph_vector(token))
    # for token in datapoint.features.previous_token:
      # prev_token_vectors.extend(self._morph_vector(token))
    # features.feature["next_token_morph_ids"].float_list.CopyFrom(self._float_feature(next_token_vectors))
    # features.feature["prev_token_morph_ids"].float_list.CopyFrom(self._float_feature(prev_token_vectors))


  def make_tf_examples(self, *, datapoints) -> List[Example]:
    tf_examples = []
    for datapoint in datapoints.datapoint:
      tf_examples.append(self.make_tf_example(datapoint))
    return tf_examples

  def _morph_vector(self, token):
    morph_vector = np.zeros(len(self.feature_mappings["morph"]), dtype=float)
    tags = morph_tags.from_token(token=token)
    morph_indices = self.numericalize(values=tags, mapping=self.feature_mappings["morph"])
    np.put(morph_vector, morph_indices, [1])
    return list(morph_vector)

  def _morph_feature(self, morphology, morph):
    """Returns case, voice, person or verbform features"""
    if morph == "case":
      morph_dict = _CASE_VALUES
    elif morph == "voice":
      morph_dict = _VOICE_VALUES
    elif morph == "person":
      morph_dict = _PERSON_VALUES
    elif morph == "verbform":
      morph_dict = _VERBFORM_VALUES
    else:
      raise ValueError(f"Cannot extract morph feature for {morph}")
    for morpheme in morphology:
      if morpheme.name == morph:
        one_hot = tf.one_hot(indices=morph_dict[morpheme.value], depth=len(morph_dict))
        return tf.keras.backend.get_value(one_hot)


  @staticmethod
  def _float_feature(values):
    float_list = tf.train.FloatList()
    float_list.value.extend(values)
    return float_list

  @staticmethod
  def _bytes_feature(values):
    bytes_list = tf.train.BytesList()
    bytes_list.value.extend(values)
    return bytes_list

  @staticmethod
  def _int_feature(values):
    int_list = tf.train.Int64List()
    int_list.value.extend(values)
    return int_list


def main(data):
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  ranker_prep = RankerPreprocessor(word_embeddings=word_embeddings, data_path=data)
  ranker_prep.make_tf_examples(datapoints=ranker_prep.train_data)


if __name__ == "__main__":
  data = "train_data.pbtxt"
  main(data)