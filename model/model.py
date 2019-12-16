# -*- coding: utf-8 -*-

"""The model class provides an interface to interact with pretrained models.

A model is a json formatted dictionary which has the following keys and values:

train_data: the string path of the training data.
test_data: the string path of the test data.
accuracy: dictionary of {test_unavg: accuracy, test_avg: accuracy}. 
nr_epochs: the number of epochs used in training.
featureset: the featureset used in training the model, in featureset_pb2 proto format.
feature_count: the number of features in the model.

"""

import os
import csv
import json
import pandas as pd
import argparse 
from collections import defaultdict
from learner import featureset_pb2
from google.protobuf import text_format
from google.protobuf import json_format
from util import reader
from util import common
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_MODEL_DIR = "model/pretrained"

class Model(object):
  """A model object is a pretrained labeler or parser model."""
  
  def __init__(self, name):
    self.name = name + ".json" if not name.endswith(".json") else name
    self.model = self._load_model(self.name)

  def get_value(self, key):
    """Returns the value of a specific key for the model.
    Args:
      key: a model key.
    Returns:
      the value for the key.
    """
    assert key in self.model.keys(), "Model has no attribute: {}".format(key)
    if key == "featureset":
      return json_format.Parse(self.model["featureset"], featureset_pb2.FeatureSet())
    return self.model[key]
  
  def write_features_as_tsv(self):
    output = os.path.join(MODEL_EXP_DIR, name.strip(".json")+"_features.tsv")
    fieldnames = ["feature_name", "feature_value", "feature_weight"]
    with open(os.path.join(output), "a") as tsvfile:
      writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
      if not file_exists:
        writer.writeheader()
      featureset = self.get_value("featureset")
      for feat in featureset:
        writer.writerow({
          "feature_name": feat.name.encode("utf-8"),
          "feature_value": feat.value.encode("utf-8"),
          "feature_weight": feat.weight,
        })
  
  def feature_impact(self):
    """Measures the impact of each feature. 
    
    We look at the sum of each feature instantiating with a different value. Then sort
    each feature based on impact.
    """
    impact = defaultdict(float)
    featureset = self.get_value("featureset")
    for feat in featureset.feature:
      impact[feat.name] += abs(round(feat.weight, 2))
    impact_df = pd.DataFrame.from_dict(impact, orient='index', columns=["impact"])
    print(impact_df.sort_values(by="impact", ascending=False, inplace=True))
    return impact
  
  def prune(self):
    featureset = self.get_value("featureset")
    pruned_featureset = featureset_pb2.FeatureSet()
    zeros = []
    for feat in featureset.feature:
      if feat.weight == 0:
        zeros.append(feat)
      else:
        pruned_featureset.feature.add(name=feat.name, value=feat.value, weight=feat.weight)
    assert len(pruned_featureset.feature) == len(featureset.feature) - len(zeros), "Pruning couldn't be validated!!"
    logging.info("ratio of zero weight features to all features: {}".format(float(len(zeros)) / float(len(featureset.feature))))
    self.model["featureset"] = pruned_featureset
    logging.info("Replaced features with pruned ones")
    
  def print_features(self):
    featureset = self.get_value("featureset")
    for feat in featureset.feature:
      print(feat.name, feat.value, feat.weight)
  
  def save(self, name):
    name = name + ".json" if not name.endswith(".json") else name
    output_file = os.path.join(_MODEL_DIR, "{}".format(name))
    model = {
        "train_data_path": self.model["train_data_path"],
        "test_data_path": self.model["test_data_path"],
        "epochs_trained": self.model["epochs_trained"],
        "test_accuracy": self.model["test_accuracy"],
        "featureset": json_format.MessageToJson(self.model["featureset"],
        	including_default_value_fields=True)
    }
    with open(output_file, "w") as output:
        json.dump(model, output, indent=4)
    logging.info("""Saved model with the following specs:
        train_data: {},
        test_data: {},
        epochs: {},
        test_accuracy: {},
        feature_count: {}""".format(
          model["train_data_path"],
          model["test_data_path"],
          model["test_accuracy"],
          model["epochs_trained"],
          len(self.model["featureset"].feature)
          )
        )
  
  def __str__(self):
    return (
      "model_name {}\n".format(self.name) + 
      "train_data: {}\n".format(self.model["train_data_path"]) +
      "test_data: {}\n".format(self.model["test_data_path"]) + 
      "test_accuracy: {}\n".format(self.model["test_accuracy"]) +
      "nr_epochs: {}\n".format(self.model["epochs_trained"]) + 
      "feature_count: {}".format(len(self.get_value("featureset").feature)))

  def _load_model(self, name):
    """Loads a model from memory.
    
    Args:
      name: the name of the model to load.
    Returns:
      model: a json formatted dictionary.
    """    
    input_file = os.path.join(_MODEL_DIR, "{}".format(name))
    with open(input_file, "r") as inp:
      model = json.load(inp)
    return model
  
  def main(args):
    model = Model(args.name)
    if args.print_feature_impact:
      model.feature_impact()
    if args.write_feature_tsv:
      model.write_features_as_tsv()
    if args.prune:
      model.prune()
    if args.output:
      model.save(name=args.output)
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, help="name of the model.")
  parser.add_argument("--output", type=str, help="path to output file you want to save the model after pruning",
                      default=None)
  parser.add_argument("--prune", type=bool, help="whether to prune the model")
  parser.add_argument("--write_feature_tsv", type=bool, help="whether to write the features as tsv file")
  parser.add_argument("--print_feature_impact", type=bool, help="whether to print the feature impacts")
  args = parser.parse_args()
  main(args)