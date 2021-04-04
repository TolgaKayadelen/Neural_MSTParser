# -*- coding: utf-8 -*-

"""Simply script to sneak peek into a pretrained model properties."""

import argparse
import json
import os

from learner import featureset_pb2
from google.protobuf import text_format
from google.protobuf import json_format
from util import common

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_model_dir = "model/pretrained/parser"

# type: model util
def GetModel(name, top_features=None):
    """Loads a pretrained model and reports its properties.

    Additionally, this function can return the top n features set up in the
    top_features parameter and report them if top_features is not None.

    Args:
        model: json, the pretrained model to load.
        top_features: int, the most important features of the model.
    Returns:
        model: the loaded model.
        top_features: the most important features.
    """
    name = name + ".json" if not name.endswith(".json") else name
    input_file = os.path.join(_model_dir, "{}".format(name))
    with open(input_file, "r") as inp:
        model = json.load(inp)
    featureset = json_format.Parse(model["featureset"], featureset_pb2.FeatureSet())
    accuracy = model["test_accuracy"]
    #feature_count = model["feature_count"]
    logging.info("Arc accuracy: {}".format(accuracy))
    #logging.info("Total number of features: {}".format(feature_count))
    logging.info("Epochs: {}".format(model["epochs_trained"]))
    logging.info("Feature Count: {}".format(len(featureset.feature)))
    logging.info("Trained on: {}".format(model["train_data_path"]))
    logging.info("Test data: {}".format(model["test_data_path"]))
    if top_features is not None:
        top = common.TopFeatures(featureset, top_features)
        for t in top:
            print(text_format.MessageToString(t, as_utf8=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to load.")
    parser.add_argument("--top_features", type=int, default=None,
        help="Top n features")
    args = parser.parse_args()
    GetModel(args.model, args.top_features)
