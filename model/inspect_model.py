# -*- coding: utf-8 -*-

"""Script to inspect the features in a pretrained model."""

import os
import csv
import json
import argparse 
from util import common
from learner import featureset_pb2
from google.protobuf import text_format
from google.protobuf import json_format
from util import reader
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_MODEL_DIR = "model/pretrained"

def inspect_model(name, top_features, output):
  # load the model
  logging.info("Loading {}".format(name))
  name = name + ".json" if not name.endswith(".json") else name
  input_file = os.path.join(_MODEL_DIR, "{}".format(name))
  with open(input_file, "r") as inp:
      model = json.load(inp)
  featureset = json_format.Parse(model["featureset"], featureset_pb2.FeatureSet())
  accuracy = model["test_accuracy"]
  logging.info("Arc accuracy of the loaded model: {}".format(accuracy))
  
  # look at the top features
  #logging.info("Getting {} top features".format(top_features))
  #top_features = common.TopFeatures(featureset, top_features)
  
  #map(lambda feat: print(feat.name, feat.value, feat.weight)), top_features)
  #logging.info("Top features..")
  #print_features(top_features)
  
  sorted_features = common.SortFeatures(featureset)  
  zero_features = [feat for feat in featureset.feature if feat.weight==0]
  #print_features(zero_features)
  total_zero_features = float(len(zero_features))
  total_features = float(len(featureset.feature))
  print("Number of features having 0 value: {}".format(total_zero_features))
  print("total features {}".format(total_features))
  print("ratio zero/total = {}".format(total_zero_features/total_features))
  write_features(name.strip(".json"), sorted_features)
  

def write_features(name, features):
  """Writes the features to a tsv file.
  
  Args:
    name: the name of the model for which we are writing the features.
    features: the features to write.
  """
  MODEL_EXP_DIR = "model/pretrained"
  file_ = os.path.join(MODEL_EXP_DIR, name+"_features.tsv")
  file_exists = os.path.isfile(file_)
  with open(os.path.join(file_), "a") as tsvfile:
    fieldnames = ["feature_name", "feature_value", "feature_weight"]
    writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
    if not file_exists:
      writer.writeheader()
    for feat in features.feature:
      writer.writerow({
        "feature_name": feat.name.encode("utf-8"),
        "feature_value": feat.value.encode("utf-8"),
        "feature_weight": feat.weight,
      })

def print_features(features):
  for feat in features:
    print(feat.name, feat.value, feat.weight)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, help="name of the model.")
  parser.add_argument("--top_features", type=int, help="the number of top features you want to check.",
                      default=100)
  parser.add_argument("--output", type=str, help="path to output file you want to save the features",
                      default=None)
  args = parser.parse_args()
  inspect_model(args.name, args.top_features, args.output)