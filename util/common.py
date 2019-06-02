# -*- coding: utf-8 -*-

"""Common utility functions."""
#TODO: unit tests for all functions here. 


from copy import deepcopy
from data.treebank import sentence_pb2
from learner import featureset_pb2
from util import reader

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
    start_token = sentence_pb2.Token(word="START_TOK", lemma="START_TOK", category="START_CAT", pos="START_POS", index=-1)
    end_token = sentence_pb2.Token(word="END_TOK", lemma="END_TOK", category="END_CAT", pos="END_POS", index=-2)
    tokens = deepcopy(sentence.token)
    del sentence.token[:]
    sentence.token.extend([start_token])
    for token in tokens:
        sentence.token.extend([token])
    sentence.token.extend([end_token])
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
        if i.word != j.word and i.word.lower() != "root"
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
    logging.info("Dropping dummy START and END tokens.")
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

# type: token util
def GetValue(token, feat):
    """Returns the desired value for a feature of a token.
    
    Args:
        token: sentence_pb2.Token()
        feat: string, the feature for which we want the value. 
       
    Returns: the desired value for a feature of the token
    """
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
                print("name: {}\n value: {}\n wta: {}\n".format(
                    name, value, weights[name][value])
                )
        print("***---------***")
    else:
        for f in features.feature:
            print("f.name: {}\n f.value: {}\n f.weight: {}\n".format(
                f.name, f.value, weights[f.name][f.value])
                )
            print("***---------***")    


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
def SortFeatures(featureset):
    #print("Sorting FeatureSet proto..")
    unsorted_list = []
    for feature in featureset.feature:
        unsorted_list.append(feature)
    sorted_list = sorted(unsorted_list, key=lambda f: f.weight, reverse=True)
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


if __name__ == "__main__":
    sentence = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
    token = sentence.token[2] # saw
    assert token.word == "saw"
    rightmost = GetRightMostChild(sentence, token)
    print("rightmost child is {}".format(rightmost.word))
                