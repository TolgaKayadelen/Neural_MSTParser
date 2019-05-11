# -*- coding: utf-8 -*-

"""Feature Extractor for the Neural MST Dependency Parser.

Takes two tokens in a sentence and extracts features for them. 

Feature representation is inspired from:
McDonald, R., Crammer, K., Pereira, F. (2006). Online Large-Margin Training of Dependency Parsers. 

TODO: Add lemma and morphology features.
"""
import re
import sys
import argparse

from google.protobuf import text_format
from util import reader
from util import common
from mst.max_span_tree import GetTokenByAddressAlt
from data.treebank import sentence_pb2 
from collections import OrderedDict

class FeatureExtractor:
    """A ruleset is a dictionary of feature extraction rules."""
    
    def __init__(self, filename, postag_window=5, log_distance=1.5):
        """Initialize a feature set from a given filename.
        
        Args:
            filename = the name of the file to read the requested features from. 
            postag_window = include a feature for each word pair that contains this many POS tags. 
            log_distance = if not None, "bin" the distance between mod and head, otherwise use 
                 linear distance. 
        
        """
        self._featureset = OrderedDict()
        self._postag_window = postag_window
        self._log_distance = log_distance
        
        for i, l in enumerate(open(filename, "r")):
            l = l.split('#')[0].split('//')[0].strip()
            #print("l is: {}".format(l))
            if not l:
                continue
            try:
                 self._featureset[l.lower()] = None
            except:
                raise
                logging.critical("%s line %d: cannot parse rule: %r", filename, i, l)
                sys.exit(1)
    
    def GetFeatures(self, sentence, head, child, use_tree_features=True):
        """Return features requested in the featureset for two tokens.
        
        Args:
            sentence: sentence_pb2.Sentence object.
            head: sentence_pb2.Token, the head token.
            child: sentence_pb2.Token, the child token.
            use_tree_features: if True, uses the head and child features of the token as well. 
        Returns:
            featureset: dict, a dictionanty of feature names and values. 
        """
        #TODO: implement the case where use_tree_features = False.
        #print("head is: {}', child is: {}'".format(head.word.encode("utf-8"), child.word.encode("utf-8")))
        for key in self._featureset:
            feature = [t.strip() for t in re.split(r'[\.\s\[\]\(\)]+', key.strip()) if t.strip()]
            value = self._GetFeatureValue(feature, 
                sentence=sentence, 
                child=child, 
                head=head,
                use_tree_features = use_tree_features)
            self._featureset[key] = value
        #for k, v in self._featureset.items():
        #    print k, v
        return self._featureset
    
    def _GetFeatureValue(self, feature, sentence=None, child=None, head=None, use_tree_features=True):
        """Get Feature Value. 
        Args:
            feature: the feature whose value we are interested in.
            child = sentence_pb2.Token(), the child token.
            head = sentence_pb2.Token(), the head token.
            sentence = sentence_pb2.Sentence()
        Returns:
            value: string, the value for the requested feature. 
        """
        #TODO: ensure this works correctly when there's more than one token btw head and child.
        feature = [f for f in feature if f != "+"]
        #print("feature is: {}".format(feature))
        value = []
        for subfeat in feature:
            subfeat = subfeat.split("_")
            offset = int(subfeat[1])
            is_tree_feature = subfeat[2] in ["up", "down"]
            is_between_feature = subfeat[0] == "between"
            t = [child, head][subfeat[0] == "head"]
            if is_tree_feature:
                if subfeat[2] == "up":
                    head_token = GetTokenByAddressAlt(
                        sentence.token,
                        sentence.token[t.index+offset].selected_head.address
                        )
                    value.append(common.GetValue(head_token, subfeat[-1]))
                elif subfeat[2] == "down":
                    child_token = common.GetRightMostChild(sentence, t)
                    value.append(common.GetValue(child_token, subfeat[-1]))
            elif is_between_feature:
                for btw_token in common.GetBetweenTokens(sentence, head, child):
                    value.append(common.GetValue(btw_token, subfeat[-1]))
            else:
                value.append(common.GetValue(sentence.token[t.index+offset], subfeat[-1]))
        return "_".join(value)
                    

def main(args):
    extractor = FeatureExtractor(filename=args.featureset_file)
    test_sentence = reader.ReadSentenceProto("./data/treebank/sentence_4.protobuf")
    head = test_sentence.token[3]
    child = test_sentence.token[1]
    features = extractor.GetFeatures(test_sentence, head, child, use_tree_features=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featureset_file", help="The file to read the feature rules from")
    args = parser.parse_args()   
    main(args)
