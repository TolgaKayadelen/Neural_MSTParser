# -*- coding: utf-8 -*-

"""Common utility functions."""

import sys
import os
import datasets
from copy import deepcopy
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from util import reader, writer
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


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
def GetChildren(sentence, token_index, children=[]):
  """Returns all the children of a token in sentence as a list.
  Args:
    sentence_pb2.Sentence()
    token_index = the index of the token in the sentence.
  Returns:
    list of children for the token at index.
  """
  for token in sentence.token:
    if token.selected_head.address == token_index:
      # print(token.index)
      # input("..")
      children.append(token)
      GetChildren(sentence, token.index, children)
  return children


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


#type: sentence util
def GetMaxlenSentence(sentences):
  """Returns the maxlen value from a set of sentences.
  Args:
    sentences: list of sentence_pb2.Sentence objects.
  Returns:  
    maxlen: int, value of maximum length sentence.
    """
  maxlen = 0
  for sentence in sentences:
    if len(sentence.token) > maxlen:
      maxlen = len(sentence.token)
  return maxlen

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

# type: feature util
def GetDistanceValue(head_index, child_index):
  """Returns the distance between head and the child tokens in the sentence.
  
  Args:
    head_index: index of the head token.
    child_index: index of the child token.
  """
  dist = head_index - child_index
  if abs(dist) <= 2:
    return "<=2"
  elif abs(dist) <= 4:
    return "<=4"
  elif abs(dist) <= 6:
    return "<=6"
  elif abs(dist) <= 8:
    return "<=8"
  else:
    return "9+"

# type: token util
def GetValue(token, feat):
    """Returns the desired value for a feature of a token.

    Args:
        token: sentence_pb2.Token()
        feat: string, the feature for which we want the value.

    Returns: the desired value for a feature of the token
    """
    morph_features = ["case", "number", "number[psor]", "person", "person[psor]", "voice", "verbform"]
    morph_value = "None"
    if token == None:
        return "None"
    elif token.index == 0 or token.pos == "TOP":
        return "ROOT"
    else:
        if feat == "word":
            return token.word
        if feat == "category":
            return token.category
        if feat == "pos":
            return token.pos
        if feat == "lemma":
            return token.lemma
        if feat in morph_features: # check whether the request is valid.
            if not len(token.morphology):
                return morph_value
            morphology = token.morphology
            for morph_feat in morphology:
                if morph_feat.name == feat:
                    morph_value = morph_feat.value
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
def ValidateAveragingArcPer(unaveraged, accumulated, averaged, iters):
    """Checks that averaging is valid and correct for the arc perceptron."""
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

def ValidateAveragingLabelPer(unaveraged, accumulated, averaged, iters):
  """Checks that averaging is valid and correct for the label perceptron."""
  for class_ in unaveraged.keys():
    for feat_name in unaveraged[class_].keys():
      for feat_val in unaveraged[class_][feat_name].keys():
        assert(averaged[class_][feat_name][feat_val] == 
          accumulated[class_][feat_name][feat_val] / iters)

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
def TopFeatures(featureset, n):
    """Return the n features with the largest weight."""
    # sort the features
    logging.info("Sorting features..")
    featureset = SortFeatures(featureset)
    if n < 0:
      # get tail features.
      return featureset.feature[n:]
    else:
      # get head features.
      return featureset.feature[:n]


def CopySentIds():
    """Copy sent ids from hf dataset examples to pbtxt files."""
    propbank_path = "./data/propbank/ud"
    write_paths = {}
    write_paths["train"] = os.path.join(propbank_path, "prp_train_with_spans_sentid.pbtxt")
    write_paths["validation"] = os.path.join(propbank_path, "prp_dev_with_spans_sentid.pbtxt")
    write_paths["test"] = os.path.join(propbank_path, "prp_test_with_spans_sentid.pbtxt")

    treebanks = {}
    dataset = datasets.load_dataset("./transformer/hf/dataset/tr/propbank_data_loader.py")
    treebanks["train"] = reader.ReadTreebankTextProto("./data/propbank/ud/prp_train_with_spans.pbtxt")
    treebanks["validation"] = reader.ReadTreebankTextProto("./data/propbank/ud/prp_dev_with_spans.pbtxt")
    treebanks["test"] = reader.ReadTreebankTextProto("./data/propbank/ud/prp_test_with_spans.pbtxt")
    sent_id_treebanks = {}
    sent_id_treebanks["train"] = treebank_pb2.Treebank()
    sent_id_treebanks["validation"] = treebank_pb2.Treebank()
    sent_id_treebanks["test"] = treebank_pb2.Treebank()
    for datatype in ["train", "validation", "test"]:
        counter = 0
        for example, sentence in zip(dataset[datatype], treebanks[datatype].sentence):
            counter += 1
            print("sentence ", counter)
            example_words = example["tokens"]
            print(example_words)
            # input()
            sentence_words = [token.word for token in sentence.token[1:]]
            print(sentence_words)
            assert(example_words == sentence_words), "Mismatch in sentences"
            sentence.sent_id = example["sent_id"]
            new_sentence = sent_id_treebanks[datatype].sentence.add()
            new_sentence.CopyFrom(sentence)
        writer.write_proto_as_text(sent_id_treebanks[datatype], write_paths[datatype])
        # input()

def CopySrlToTokens():
    treebanks = {}
    output_treebanks = {}
    write_paths = {}
    treebanks["train"] = reader.ReadTreebankTextProto(
        "./data/propbank/ud/without_spans/prp_train_without_spans_sentid.pbtxt")
    treebanks["dev"] = reader.ReadTreebankTextProto(
        "./data/propbank/ud/without_spans/prp_dev_without_spans_sentid.pbtxt")
    treebanks["test"] = reader.ReadTreebankTextProto(
        "./data/propbank/ud/without_spans/prp_test_without_spans_sentid.pbtxt")
    output_treebanks["train"] = treebank_pb2.Treebank()
    output_treebanks["dev"] = treebank_pb2.Treebank()
    output_treebanks["test"] = treebank_pb2.Treebank()
    write_paths["train"] = "./data/propbank/ud/srl/train.pbtxt"
    write_paths["dev"] = "./data/propbank/ud/srl/dev.pbtxt"
    write_paths["test"] = "./data/propbank/ud/srl/test.pbtxt"
    for datatype in ["train", "dev", "test"]:
        sentences = treebanks[datatype].sentence
        output_treebank = output_treebanks[datatype]
        for sentence in sentences:
            tokens = sentence.token
            for token in tokens:
                token.srl = "-0-"
                token.predicative = 0
            for arg_str in sentence.argument_structure:
                tokens[arg_str.predicate_index].predicative = 1
                for argument in arg_str.argument:
                    token_index = argument.token_index[0]

                    tokens[token_index].srl = argument.srl
            new_sentence = output_treebank.sentence.add()
            new_sentence.CopyFrom(sentence)
        writer.write_proto_as_text(output_treebank, write_paths[datatype])



if __name__ == "__main__":
    CopySrlToTokens()