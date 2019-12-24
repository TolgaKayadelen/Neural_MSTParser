# -*- coding: utf-8 -*-

"""Feature Extractor for the Dependency Parser and Dependency Labeler.

Feature representation is inspired from:
McDonald, R., Crammer, K., Pereira, F. (2006). Online Large-Margin Training of Dependency Parsers.

"""
import argparse
import os
import re
import sys

from collections import OrderedDict
from google.protobuf import text_format
from util import reader
from util import common
from data.treebank import sentence_pb2
from learner import featureset_pb2

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

FEATURE_DIR = "learner/features"
TEST_FEATURE_DIR = "data/testdata/features/test_features"

class FeatureExtractor:

    def __init__(self, featuretype, feature_file, test=False):
        """Initialize this feature extractor with a featureset file.

        Args:
          featuretype: either labelfeatures or arcfeatures.
          feature_file: the file from which to read the features.
          test: If True, it means the extractor is initialized for test purposes only, and reads
            data from the TEST_FEATURE_DIR.

        """
        assert featuretype in ["arcfeatures", "labelfeatures"], "Invalid feature type!!"
        if featuretype == "arcfeatures" and not test:
          self._feature_file = os.path.join(FEATURE_DIR, "{}.txt".format(feature_file))
          logging.info("reading arc features from {}".format(self._feature_file))
        elif featuretype == "labelfeatures" and not test:
          self._feature_file = os.path.join(FEATURE_DIR, "{}.txt".format(feature_file))
          logging.info("reading label features from {}".format(self._feature_file))
        elif featuretype == "arcfeatures" and test:
          self._feature_file = os.path.join(TEST_FEATURE_DIR, "arcfeatures.txt")
          logging.info("reading arc features from {}".format(self._feature_file))
        else:
          self._feature_file = os.path.join(TEST_FEATURE_DIR, "labelfeatures.txt")
          logging.info("reading label features from {}".format(self._feature_file))

    def InitializeFeatureNames(self):
        """Initialize a featureset proto with the feature names read from featureset file.

        Returns:
            featureset: featureset_pb2.FeatureSet(), only has feature_names.
        """
        featureset = featureset_pb2.FeatureSet()
        for i, l in enumerate(open(self._feature_file, "r")):
            l = l.split('#')[0].split('//')[0].strip()
            #print("l is: {}".format(l))
            if not l:
                continue
            try:
                 # Initialize the featureset vector only with feature names.
                 featureset.feature.add(name=l.lower())
            except:
                raise
                logging.critical("%s line %d: cannot parse rule: %r", self._feature_file, i, l)
                sys.exit(1)
        return featureset

    def GetFeatures(self, sentence, head, child):
        """Return the values for the features in the featureset for head and child tokens.

        Args:
            sentence: sentence_pb2.Sentence object.
            head: sentence_pb2.Token, the head token.
            child: sentence_pb2.Token, the child token.
        Returns:
            featureset: featureset_pb2.FeatureSet(), a proto of feature names and values.
            Note that this doesn't return any weight for the features.
        """
        #print("head is: {}, child is: {}".format(head.word.encode("utf-8"), child.word.encode("utf-8")))
        featureset = self.InitializeFeatureNames()
        for feature in featureset.feature:
            #feature = [t.strip() for t in re.split(r'[\.\s\[\]\(\)]+', key.strip()) if t.strip()]
            feats = [t.strip() for t in feature.name.split("+")]
            #print("feature {}".format(feats))
            value = self._GetFeatureValue(feats,
                sentence=sentence,
                head=head,
                child=child)
            feature.value = value
        #common.PPrintTextProto(featureset)
        return featureset

    def _GetFeatureValue(self, feature, sentence=None, head=None, child=None):
        """Get Feature Value.
        Args:
            feature: the feature whose value we are interested in.
            child = sentence_pb2.Token(), the child token.
            head = sentence_pb2.Token(), the head token.
            sentence = sentence_pb2.Sentence()
        Returns:
            value: string, the value for the requested feature.
        """
        feature = [f for f in feature if f != "+"]
        #print("feature is: {}".format(feature))
        value = []
        for subfeat in feature:
            subfeat = subfeat.split("_")
            #print(subfeat)
            if subfeat[0] == "distance":
              is_distance_feature = True
            else:
              is_distance_feature = False
              offset = int(subfeat[1])
            is_between_feature = subfeat[0] == "between"
            is_tree_feature = subfeat[2] in ["up", "down"]
            t = [child, head][subfeat[0] == "head"] # t = token.
            dummy_start_token = 1 if sentence.token[0].word == "START_TOK" else 0
            #print(sentence.token[t.index+offset+dummy_start_token])
            if is_tree_feature and not t.index == 0:
              if subfeat[2] == "up":
                head_of_t = common.GetTokenByAddress(
                  sentence.token,
                  sentence.token[t.index+offset+dummy_start_token].selected_head.address
                )
                value.append(common.GetValue(head_of_t, subfeat[-1]))
              elif subfeat[2] == "down":
                child_of_t = common.GetRightMostChild(sentence, t)
                value.append(common.GetValue(child_of_t, subfeat[-1]))
            elif is_between_feature:
                for btw_token in common.GetBetweenTokens(sentence, head, child, dummy_start_token):
                    value.append(common.GetValue(btw_token, subfeat[-1]))
            elif is_distance_feature:
              value.append(common.GetDistanceValue(head.index, child.index))
            else:
                if not t.index+offset+dummy_start_token >= len(sentence.token):
                  value.append(common.GetValue(sentence.token[t.index+offset+dummy_start_token], subfeat[-1]))
                else:
                  value.append("None")
        #print("value {}".format(value))
        return "_".join(value)


def main(args):
    extractor = FeatureExtractor(featuretype=args.featuretype)
    test_sentence = reader.ReadSentenceTextProto(
        "./data/testdata/common/john_saw_mary_extended.pbtxt")
    head = test_sentence.token[3]
    child = test_sentence.token[1]
    features = extractor.GetFeatures(test_sentence, head, child)
    print(text_format.MessageToString(features, as_utf8=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featuretype", help="The type of features.")
    args = parser.parse_args()
    main(args)
