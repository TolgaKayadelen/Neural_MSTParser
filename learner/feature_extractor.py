# -*- coding: utf-8 -*-

"""Feature Extractor for the Neural MST Dependency Parser.

Takes two tokens in a sentence and extracts features for them.

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

FEATURE_DIR = "learner/features"

class FeatureExtractor:

    def __init__(self, featuretype, postag_window=5, log_distance=1.5):
        """Initialize this feature extractor with a featureset file.

        Args:
            postag_window = include a feature for each word pair that contains this many POS tags.
            log_distance = if not None, "bin" the distance between mod and head, otherwise use
                 linear distance.

        """
        assert featuretype in ["arcfeatures", "labelfeatures"], "Invalid feature type!!"
        #TODO: create a data dependency for it to be available in unit tests.
        if featuretype == "arcfeatures":
            self._feature_file = os.path.join(FEATURE_DIR, "arcfeatures.txt")
            #self._feature_file = "./learner/arcfeatures.txt"
        else:
            self._feature_file = os.path.join(FEATURE_DIR, "labelfeatures.txt")
        self._postag_window = postag_window
        self._log_distance = log_distance

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
        #print("head is: {}', child is: {}'".format(head.word.encode("utf-8"), child.word.encode("utf-8")))
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
        #TODO: ensure this method works correctly when more than one token btw head and child.
        feature = [f for f in feature if f != "+"]
        #print("feature is: {}".format(feature))
        value = []
        for subfeat in feature:
            subfeat = subfeat.split("_")
            offset = int(subfeat[1])
            is_between_feature = subfeat[0] == "between"
            t = [child, head][subfeat[0] == "head"] # t = token.
            dummy_start_token = 1 if sentence.token[0].word == "START_TOK" else 0
            if is_between_feature:
                for btw_token in common.GetBetweenTokens(sentence, head, child, dummy_start_token):
                    value.append(common.GetValue(btw_token, subfeat[-1]))
            else:
                value.append(common.GetValue(sentence.token[t.index+offset+dummy_start_token], subfeat[-1]))
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
