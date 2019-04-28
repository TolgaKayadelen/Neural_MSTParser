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
from mst.max_span_tree import GetTokenByAddressAlt
from data.treebank import sentence_pb2 
from collections import OrderedDict

class FeatureSet:
    """A ruleset is a dictionary of feature extraction rules."""
    
    def __init__(self, filename, postag_window = 5, log_distance = 1.5):
        """Initialize a feature set from a given filename.
        
        Args:
            filename = the name of the file to read the features from. 
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
                 #self.append(Rule(l.lower()))
            except:
                raise
                logging.critical("%s line %d: cannot parse rule: %r", filename, i, l)
                sys.exit(1)
        self.GetFeatures()
    
    
    def GetFeatures(self, sentence=None, head=None, child=None, use_tree_features=True):
        test_sentence = reader.ReadSentenceProto("./data/treebank/sentence_4.protobuf")
        #print(text_format.MessageToString(test_sentence, as_utf8=True))
        #print(sentence)
        child = test_sentence.token[1]
        head = test_sentence.token[3]
        print("head is: {}', child is: {}'".format(head.word.encode("utf-8"), child.word.encode("utf-8")))
        for key in self._featureset:
            feature = [t.strip() for t in re.split(r'[\.\s\[\]\(\)]+', key.strip()) if t.strip()]
            value = self._GetFeatureValue(feature, sentence=test_sentence, child=child, head=head)
            self._featureset[key] = value
        for k, v in self._featureset.items():
            print k, v
    
    def _GetFeatureValue(self, feature, sentence=None, child=None, head=None, use_tree_features=True):
        feature = [t for t in feature if t != "+"]
        #print("feature is: {}".format(feature))
        value = []
        for subfeat in feature:
            subfeat = subfeat.split("_")
            offset = int(subfeat[1])
            is_tree_feature = subfeat[2] in ["up", "down"]
            t = [child, head][subfeat[0] == "head"]
            if not is_tree_feature:
                if subfeat[-1] == "word":
                    #print sentence.token[t.index+offset].word.encode("utf-8")
                    value.append(sentence.token[t.index+offset].word.encode("utf-8"))
                elif subfeat[-1] == "lemma":
                    #print sentence.token[t.index+offset].lemma.encode("utf-8")
                    value.append(sentence.token[t.index+offset].lemma.encode("utf-8"))
                elif subfeat[-1] == "pos":
                    #print sentence.token[t.index+offset].pos.encode("utf-8")
                    value.append(sentence.token[t.index+offset].pos.encode("utf-8"))
            else:
                if subfeat[2] == "up":
                    head_token = GetTokenByAddressAlt(
                        sentence.token,
                        sentence.token[t.index+offset].selected_head.address
                        )
                    if subfeat[-1] == "word":
                        value.append(head_token.word.encode("utf-8"))
                    elif subfeat[-1] == "lemma":
                        value.append(head_token.lemma.encode("utf-8"))
                    elif subfeat[-1] == "pos":
                        value.append(head_token.pos.encode("utf-8"))
                elif subfeat[2] == "down":
                    pass
                    #TODO: child_token = common.GetRightmostChild(sentence.token, t)
        return "_".join(value)
            #if subfeat[0] == "head":
            #    if subfeat[-1] == "word":
            #        print sentence.token[head.index+offset].word.encode("utf-8")
            #    elif subfeat[-1] == "pos":
            #        print sentence.token[head.index+offset].pos.encode("utf-8")
            #else:
            #    pass
        
    
            

class Rule:
    """A rule describes how to extract features from a sentence."""
    
    def __init__(self, raw):
        """Initialize with a raw rule decsription string."""
        names = []
        subrules = []
        #for chunk in raw.split("+"):
        #    r, n = self._parse(chunk)
        #    subrules.append(r)
        #    names.append(n)
        #self._subrules = subrules
        #self._name = "+".join(names)
        #print("subrules: {}, features: {}".format(self._subrules, self._name))
    
    def apply(self, sentence, head, child, use_tree_features=True):
        pass
        
    def _parse(self, chunk):
        """Parse a chunk of subrule string."""
        tokens = [t.strip() for t in re.split(r'[\.\s\[\]\(\)]+', chunk.strip()) if t.strip()]
        print("tokens: {}".format(tokens))
        # Whether it is a word, lemma, or postag feature
        feature_type = tokens.pop()
        print("feature type: {}".format(feature_type))
        feature = self._parse_feature(feature_type)
        print("feature: {}".format(feature))
        
        start_from_head = None
        offset = None
        motions = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            print("token is {}".format(token))
            i += 1
            if token == "head":
                assert start_from_head is None
                start_from_head = True
                offset = int(tokens[i])
                i += 1
            elif token == "child":
                assert start_from_head is None
                start_from_head = False
                offset = int(tokens[i])
                i += 1
            elif token == "up":
                motions.append(MOTION_UP)
                self.is_tree_feature = True

    def _parse_feature(self, token):
        """Convert a feature name into a feature index."""
        #if token == "deprel":
        #    self.is_tree_feature = TRUE
        #    return sentence.FEATURE_LABEL
        if token in FEATURES:
            return FEATURES.index(token)
        raise NameError("unknown feature {}".format(token))
                    
def main(args):
    feature_set = FeatureSet(filename=args.featureset_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featureset_file", help="The file to read the feature rules from")
    args = parser.parse_args()   
    main(args)
