# -*- coding: utf-8 -*-

"""Common utility functions."""

import os
from copy import deepcopy
from data.treebank import sentence_pb2
from learner import featureset_pb2
from util import reader
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


#type: sentence util
def ExtendSentence(sentence):
    """Adds a dummy start and end node (tokens) to the sentence.

    Args:
        sentence: sentence_pb2.Sentence()
    Returns:
        sentence: sentence_pb2.Sentence()
    """
    sentence.length = len(sentence.token)
    start_token = sentence_pb2.Token(word="START_TOK", lemma="START_TOK", category="START_CAT", pos="START_POS", index=-1)
    end_token = sentence_pb2.Token(word="END_TOK", lemma="END_TOK", category="END_CAT", pos="END_POS", index=-2)
    tokens = deepcopy(sentence.token)
    del sentence.token[:]
    sentence.token.extend([start_token])
    for token in tokens:
        sentence.token.extend([token])
    sentence.token.extend([end_token])
    assert sentence.length == len(sentence.token) - 2, "Wrong sentence length after extension!"
    return sentence


# type: sentence util
def GetRightMostChild(sentence, token):
    """Returns the rightmost child of token in sentence.

    Args:
        token, token.pb2 object. The token whose child we want.
        sentence, sentence.pb2 object. The sentence we search to get the child of token.
    Returns:
        rightmost = the rightmost child token.
    """
    assert token.HasField("index"), "Token doesn't have index."
    children = []
    max_ = 0
    rightmost = None
    for child in sentence.token:
        #print("evalutating child {}".format(child.word))
        #print("selected head for this child {}".format(child.selected_head.address))
        if child.selected_head.address == token.index:
            if child.index > max_:
                rightmost = child
            max_ = child.index
    #return max(children, key=lambda x:x.index)
    return rightmost


#type: sentence util
def GetBetweenTokens(sentence, head, child, dummy=0):
    """Returns a list of tokens between the head and the child"""
    assert head.HasField("index") and child.HasField("index"), "Token has no index"
    if head.index > child.index:
        btw_tokens = sentence.token[child.index+1:head.index]
    else:
        btw_tokens = sentence.token[head.index+1+dummy:child.index+dummy]
    return btw_tokens if btw_tokens else [None]


# type: sentence util
def ConnectSentenceNodes(sentence):
    """Connect the tokens (nodes) of a sentence to every other token.

    Every token has every other token as a head candidate, except for itself.
    The ROOT token has no head candidates.

    Args:
        sentence: A protocol buffer Sentence object.
    Returns:
        The sentence where all tokens are connected.
    """
    token_connections = [
        (i, j)
        for i in sentence.token for j in sentence.token[::-1]
        if i.index != j.index and i.word.lower() != "root"
        ]

    for edge in token_connections:
        # ch: candidate_head
        ch = edge[0].candidate_head.add()
        # ch.address = _GetTokenIndex(sentence.token, edge[1])
        ch.address = edge[1].index
        ch.arc_score = 0.0 # default

    # Sanity checking:
    # Each token should have sentence.length - 1 number of candidate heads, i.e.
    # all other tokens except for itself. The root token (indexed 0) should not
    # have any candidate heads.
    for token in sentence.token:
        if token.index == 0:
            assert len(token.candidate_head) == 0
        else:
            #print("token {}, candidate heads: {}".format(token, len(token.candidate_head)))
            assert len(token.candidate_head) == len(sentence.token) - 1
    return sentence


#type: sentence util
def DropDummyTokens(sentence):
    """Drops the dummy START and END tokens from the sentence.

    Args:
        sentence: sentence_pb2.Sentence()
    Returns:
        sentence: sentence where the dummy tokens are dropped.
    """
    #logging.info("Dropping dummy START and END tokens.")
    if not sentence.token[0].index == -1 and sentence.token[-1].index == -2:
        assert sentence.length == len(sentence.token), """Sentence token count
            doesn't match sentence.length attribute!"""
        logging.info("The sentence doesn't have any dummy tokens.")
        return sentence
    else:
        new_s = sentence_pb2.Sentence()
        new_s.length = sentence.length
        for token in sentence.token:
            if token.index in [-1, -2] or token.word in ["START_TOK", "END_TOK"]:
                continue
            t = new_s.token.add()
            t.CopyFrom(token)
        assert new_s.length == len(new_s.token), """Sentence token count
            doesn't match sentence.length attribute!"""
        return new_s

# type: sentence util
def GetTokenByAddress(tokens, address):
    """Function to get the token in the specified address.
    Args:
        tokens: list, list of sentence_pb2.Token() objects.
        address: int, the address we are searching for.
    Returns:
        token: sentence_pb2.Token(), the token at the specified address.

    """
    list_indices = []
    for token in tokens:
        assert token.HasField("index"), "Token doesn't have index."
        list_indices.append(token.index)
        #common.PPrintTextProto(token)
        #print(list_indices)
        assert list_indices.count(token.index) == 1, "Can't have two tokens with same index."
        #print("searching for: {}, token_index: {}".format(address, str(token.index)))
        if token.index == address:
            found = token
            break
    return found

# type: sentence util
def GetSentenceWeight(sentence):
    """Returns the weight of the sentence based on arc scores.
    Args:
        sentence: sentence_pb2.Sentence()
    Returns:
        sentence_weight: float, the weight of the sentence.
    """
    sentence_weight = 0.0
    for token in sentence.token:
        sentence_weight += token.selected_head.arc_score
    return sentence_weight

# type: token util
def GetValue(token, feat):
    """Returns the desired value for a feature of a token.

    Args:
        token: sentence_pb2.Token()
        feat: string, the feature for which we want the value.

    Returns: the desired value for a feature of the token
    """
    morph_features = ["case", "number", "number[psor]"]
    morph_value = "None"
    if token == None:
        return "None"
    elif token.index == 0 or token.pos == "TOP":
        return "ROOT"
    else:
        if feat == "word":
            return token.word.encode("utf-8")
        if feat == "pos":
            return token.pos.encode("utf-8")
        if feat == "lemma":
            return token.lemma.encode("utf-8")
        if feat in morph_features: # check whether the request is valid.
            if not len(token.morphology):
                return morph_value
            morphology = token.morphology
            for morph_feat in morphology:
                if morph_feat.name == feat:
                    morph_value = morph_feat.value.encode("utf-8")
                    break
            return morph_value
        else:
            logging.error("Can't extract value for this feature: {}".format(feat))



# type: print util
def PPrintWeights(weights, features=None):
    """Utility function to pretty print weights for selected features.

    Args:
        weights: defaultdict(dict), the weights dict.
        features: featureset_pb2.FeatureSet, the featureset whose weights
            to print.
    """
    # if there are no specifically requested features,
    # just print all that's in weights.
    if not features:
        for name in weights.keys():
            for value in weights[name].keys():
                print("name: {}\n value: {}\n weight: {}\n".format(
                    name.encode("utf-8"), value.encode("utf-8"), weights[name][value])
                )
        print("***---------***")
    else:
        for f in features.feature:
            print("f.name: {}\n f.value: {}\n f.weight: {}\n".format(
                f.name, f.value, weights[f.name][f.value])
                )
            print("***---------***")

#type: print util
def PPrintTextProto(message):
    print(text_format.MessageToString(message, as_utf8=True))


# type: weights util
def CompareWeights(weights1, weights2):
    """Utility function prints comparison of two weights dicts."""
    weights1_ = featureset_pb2.FeatureSet()
    weights2_ = featureset_pb2.FeatureSet()

    for name in weights1.keys():
        for value in weights1[name].keys():
            weights1_.feature.add(name=name, value=value, weight=weights1[name][value])

    for name in weights2.keys():
        for value in weights2[name].keys():
            weights2_.feature.add(name=name, value=value, weight=weights2[name][value])

    print(zip(list(weights1_.feature), list(weights2_.feature)))

# type: weights util
def ValidateAveragedWeights(unaveraged, accumulated, averaged, iters):
    """Checks that averaging is valid and correct."""
    for name in unaveraged.keys():
        for value in unaveraged[name].keys():
            assert averaged[name][value] == accumulated[name][value] / iters

    # uncomment to see an example
    '''
    feat_name = "head_0_pos"
    feat_value = "Noun"
    unaveraged_weights_for_feat = unaveraged[feat_name][feat_value]
    accumulated_weights_for_feat = accumulated[feat_name][feat_value]
    averaged_weights_for_feat = averaged[feat_name][feat_value]
    print("feat name {}, feat value {}".format(feat_name, feat_value))
    print("unaveraged value= {}".format(unaveraged_weights_for_feat))
    print("accumulated value= {}".format(accumulated_weights_for_feat))
    print("averaged_weights_for_feat= {}".format(averaged_weights_for_feat))
    print("average is equal to {} = {} / {}".format(averaged_weights_for_feat,
                                                    accumulated_weights_for_feat,
                                                    iters))
    '''

# type: featureset proto util
def GetFeatureWeights(weights, features):
    """Utility function to return the weights for selected features.
    Args:
        weights: defaultdict(dict), the weights dict.
        features: featureset_pb2.FeaturSet, the featureset whose weights
            to return.
    Returns:
        list of weights.
    """
    return [weights[f.name][f.value] for f in features.feature]

# type: featureset proto util
def SortFeatures(featureset, sort_key=lambda f: f.weight):
    #print("Sorting FeatureSet proto..")
    unsorted_list = []
    for feature in featureset.feature:
        unsorted_list.append(feature)
    sorted_list = sorted(unsorted_list, key=sort_key, reverse=True)
    sorted_featureset = featureset_pb2.FeatureSet()
    for f in sorted_list:
        sorted_featureset.feature.add(name=f.name, value=f.value, weight=f.weight)
    del sorted_list
    del unsorted_list
    featureset.CopyFrom(sorted_featureset)
    return featureset

# type: featureset proto util
def TopFeatures(featureset, n):
    """Return the n features with the largest weight."""
    featureset = SortFeatures(featureset)
    return featureset.feature[:n]

def GetLabels():
    """Util function to read label_to_enum.tsv and return labels as a dict."""
    _MODEL_DIR = "model"
    label_to_enum = {}
    with open(os.path.join(_MODEL_DIR, "label_to_enum.tsv")) as tsvfile:
      for row in tsvfile:
        label_to_enum[row.split()[0]] = int(row.split()[1])
    return label_to_enum

if __name__ == "__main__":
    sentence = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
    token = sentence.token[2] # saw
    assert token.word == "saw"
    rightmost = GetRightMostChild(sentence, token)
    print("rightmost child is {}".format(rightmost.word))
