"""Preprocessor module that creates tf.data.Dataset for the ranker from a set of datapoints."""

import os
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
_RANKER_DATA_DIR = "./ranker/data"
_DATA_DIR = "./data/ "
_CASE_VALUES = {"abl": 0, "acc": 1, "dat": 2, "equ": 3, "gen": 4, "ins": 5, "loc": 6, "nom": 7}
_PERSON_VALUES = {"1": 0, "2": 1, "3": 2}
_VOICE_VALUES = {"cau": 0, "pass": 1, "rcp": 2, "rfl": 3}
_VERBFORM_VALUES = {"conv": 0, "part": 1, "vnoun": 2}
_MORPHOLOGY = {"case": _CASE_VALUES, "person": _PERSON_VALUES, "voice": _VOICE_VALUES, "verbform": _VERBFORM_VALUES}

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
               features: List[str] = ["words", "pos", "morph", "dep_labels"]):
    super(RankerPreprocessor, self).__init__(word_embeddings=word_embeddings,
                                             features=features,
                                             labels=[])
    self.feature_mappings = self.get_index_mappings_for_features(self.features)

  def make_tf_example(self, datapoint) -> tf.train.Example:
    # Create a tf example for each hypothesis
    tf_examples = []
    for hypothesis in datapoint.hypotheses:
      example = tf.train.Example()
      features = example.features
      features.feature["hypo_label"].bytes_list.CopyFrom(self._bytes_feature([hypothesis.label.encode("utf-8")]))
      features.feature["hypo_label_id"].int64_list.CopyFrom(self._int_feature([hypothesis.label_id]))
      features.feature["hypo_rank"].int64_list.CopyFrom(self._int_feature([hypothesis.rank]))
      features.feature["hypo_reward"].float_list.CopyFrom(self._float_feature([hypothesis.reward]))


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
      for feat in ["case", "voice", "person", "verbform"]:
        morph_vector= self._morph_feature(datapoint.features.morphology, feat)
        if morph_vector is not None:
          features.feature[feat].float_list.CopyFrom(self._float_feature(morph_vector))
        else:
          features.feature[feat].float_list.CopyFrom(self._float_feature(np.zeros(len(_MORPHOLOGY[feat]))))
      features.feature["word_string"].bytes_list.CopyFrom(self._bytes_feature([datapoint.word.encode("utf-8")]))

      # next_token_vectors, prev_token_vectors = [], []
      # for token in  datapoint.features.next_token:
      # next_token_vectors.extend(self._morph_vector(token))
      # for token in datapoint.features.previous_token:
      # prev_token_vectors.extend(self._morph_vector(token))
      # features.feature["next_token_morph_ids"].float_list.CopyFrom(self._float_feature(next_token_vectors))
      # features.feature["prev_token_morph_ids"].float_list.CopyFrom(self._float_feature(prev_token_vectors))
      # print(example)
      tf_examples.append(example)
      # input()
    return tf_examples

  def make_tf_examples(self, *, datapoints) -> List[Example]:
    tf_examples = []
    for datapoint in datapoints.datapoint:
      tf_examples.extend(self.make_tf_example(datapoint))
    return tf_examples

  # NOTE:
  #
  # We keep the batch_size 5 because each hypothesis is a feature example. Therefore, 5 feature examples
  # correspond to the label ranks for a single token.
  def make_dataset_from_generator(self, *, datapoints, batch_size=5) -> Dataset:
    hypothesis_datapoint_map = []
    for datapoint in datapoints.datapoint:
      hypothesis_datapoint_map.extend([(h, datapoint) for h in datapoint.hypotheses])
    generator = lambda : self._example_generator(hypothesis_datapoint_map)

    output_shapes = {
      "word_id": [],
      "hypo_label": [],
      "hypo_label_id": [],
      "hypo_rank": [],
      "hypo_reward": [],
      "pos_id": [],
      "case": [8],
      "person": [3],
      "voice": [4],
      "verbform": [3],
      "word_string": [],
      "next_token_ids": [None],
      "next_token_pos_ids": [None],
      "prev_token_ids": [None],
      "prev_token_pos_ids": [None],
    }
    output_types = {
      "word_id": tf.int64,
      "hypo_label": tf.string,
      "hypo_label_id": tf.int64,
      "hypo_rank": tf.int64,
      "hypo_reward": tf.float32,
      "pos_id": tf.int64,
      "case": tf.float32,
      "person": tf.float32,
      "voice": tf.float32,
      "verbform": tf.float32,
      "word_string": tf.string,
      "next_token_ids": tf.int64,
      "next_token_pos_ids": tf.int64,
      "prev_token_ids": tf.int64,
      "prev_token_pos_ids": tf.int64,
    }
    padded_shapes = {
      "word_id": [],
      "hypo_label": [],
      "hypo_label_id": [],
      "hypo_rank": [],
      "hypo_reward": [],
      "pos_id": [],
      "word_string": [],
      "next_token_ids": [2],
      "next_token_pos_ids": [2],
      "prev_token_ids": [2],
      "prev_token_pos_ids": [2],
      "case": [8],
      "person": [3],
      "voice": [4],
      "verbform": [3]
     }
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=output_shapes,
                                             output_types=output_types)

    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    # for batch in dataset:
    #   print(batch)
    #   input()
    return dataset

  def _example_generator(self, datapoints):
    print(f"Total datapoints {len(datapoints)}")
    yield_dict = {}
    for datapoint in datapoints:
      hypothesis, datapoint = datapoint
      yield_dict["word_id"]=datapoint.word_id
      yield_dict["hypo_label"]=hypothesis.label.encode("utf-8")
      yield_dict["hypo_label_id"]=hypothesis.label_id
      yield_dict["hypo_rank"]=hypothesis.rank
      yield_dict["hypo_reward"]=hypothesis.reward
      yield_dict["pos_id"]=self.numericalize(
        values=[datapoint.features.pos], mapping=self.feature_mappings["pos"]
      )[0]
      yield_dict["word_string"]=datapoint.word.encode("utf-8")
      yield_dict["next_token_ids"] = self.numericalize(
        values=[t.word for t in datapoint.features.next_token],
        mapping=self.feature_mappings["words"])
      yield_dict["prev_token_ids"] = self.numericalize(
        values=[t.word for t in datapoint.features.previous_token],
        mapping=self.feature_mappings["words"])
      yield_dict["next_token_pos_ids"]=self.numericalize(
        values=[t.pos for t in datapoint.features.next_token],
        mapping=self.feature_mappings["pos"]
      )
      yield_dict["prev_token_pos_ids"]=self.numericalize(
        values=[t.pos for t in datapoint.features.previous_token],
        mapping=self.feature_mappings["pos"]
      )
      for feat in ["case", "voice", "person", "verbform"]:
        morph_vector = self._morph_feature(datapoint.features.morphology, feat)
        if morph_vector is not None:
          yield_dict[feat] = morph_vector
        else:
          yield_dict[feat] = np.zeros(len(_MORPHOLOGY[feat]))
      # print("yield dict ", yield_dict)
      # input()
      yield yield_dict

  def _morph_vector(self, token):
    morph_vector = np.zeros(len(self.feature_mappings["morph"]), dtype=float)
    tags = morph_tags.from_token(token=token)
    morph_indices = self.numericalize(values=tags, mapping=self.feature_mappings["morph"])
    np.put(morph_vector, morph_indices, [1])
    return list(morph_vector)

  def _morph_feature(self, morphology, morph):
    """Returns case, voice, person or verbform features"""
    if morphology is None:
      return
    try:
      morph_dict = _MORPHOLOGY[morph]
    except KeyError:
      logging.error(f"Feat {morph} not found in supported morphology features.")
      return
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


def read_dataset_from_tfrecords(records: str, batch_size: int = 5) -> Dataset:

  def _parse_tf_records(record):
    var_len_features = ["next_token_ids", "next_token_pos_ids", "prev_token_ids", "prev_token_pos_ids"]
    return_dict = {}
    feature_description = {
      "word_id": tf.io.FixedLenFeature([], tf.int64),
      "hypo_label_id": tf.io.FixedLenFeature([], tf.int64),
      "hypo_rank": tf.io.FixedLenFeature([], tf.int64),
      "hypo_reward": tf.io.FixedLenFeature([], tf.float32),
      "pos_id": tf.io.FixedLenFeature([], tf.int64),
      "case": tf.io.FixedLenFeature([8], tf.float32),
      "person": tf.io.FixedLenFeature([3], tf.float32),
      "voice": tf.io.FixedLenFeature([4], tf.float32),
      "verbform": tf.io.FixedLenFeature([3], tf.float32),
      "word_string": tf.io.FixedLenFeature([], tf.string),
      "next_token_ids": tf.io.VarLenFeature(dtype=tf.int64),
      "next_token_pos_ids": tf.io.VarLenFeature(dtype=tf.int64),
      "prev_token_ids": tf.io.VarLenFeature(dtype=tf.int64),
      "prev_token_pos_ids": tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed_example = tf.io.parse_example(record, feature_description)
    for key in feature_description.keys():
      if key in var_len_features:
        return_dict[key] = tf.sparse.to_dense(parsed_example[key])
      else:
        # print("key ", key)
        return_dict[key] = parsed_example[key]
    return return_dict

  padded_shapes = {"word_id": [], "hypo_label_id": [], "hypo_rank": [], "hypo_reward": [],
                   "pos_id": [], "word_string": [], "next_token_ids": [2],
                   "next_token_pos_ids": [2], "prev_token_ids": [2], "prev_token_pos_ids": [2],
                   "case": [8],
                   "person": [3], "voice": [4],
                   "verbform": [3]
                   }

  raw_dataset = tf.data.TFRecordDataset([records])
  dataset = raw_dataset.map(_parse_tf_records).padded_batch(batch_size=batch_size,
                                                            padded_shapes=padded_shapes)
  return dataset


def main(data):
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  ranker_prep = RankerPreprocessor(word_embeddings=word_embeddings)
  # ranker_datapoints=reader.ReadRankerTextProto(os.path.join(_RANKER_DATA_DIR, data))
  # dataset = ranker_prep.make_dataset_from_generator(datapoints=ranker_datapoints)
  # tf_examples = ranker_prep.make_tf_examples(datapoints=ranker_datapoints)
  # ranker_prep.write_tf_records(examples=tf_examples, path=os.path.join(_RANKER_DATA_DIR, "tr_boun-ud-dev-ranker-data-rio"))
  # writer.write_protolist_as_text(tf_examples, path=os.path.join(_RANKER_DATA_DIR, "tr_boun-ud-dev-ranker-data-rio.pbtxt"))
  # logging.info(f"{len(tf_examples)} examples written to {_RANKER_DATA_DIR}")

  # read dataset
  dataset = read_dataset_from_tfrecords(
    records=os.path.join(_RANKER_DATA_DIR, 'tr_boun-ud-dev-ranker-data-rio.tfrecords'),
    batch_size=5)
  for batch in dataset:
    print(batch)
    input()
  # for data in dataset:
  #  print(data)
  #  input()
    # print(data["next_token_pos_ids"])

if __name__ == "__main__":
  data = "tr_boun-ud-dev-ranker-datapoint.pbtxt"
  main(data)