import os
import gensim

import numpy as np
from util import reader
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from collections import defaultdict, OrderedDict, Counter

_EMBEDDING_DIR = "embeddings"


def convert_to_one_hot(indices, n_labels):
  """Converts an integer array of shape (1, m) to a one hot vector of shape (len(indices), n_labels)
  
  Args:
    indices: An integer array representing the labels or classes.
    n_labels: depth of the one hot vector.
  Returns:
    one hot vector of shape (len(indices), n_labels)
  """
  return np.eye(n_labels)[indices.reshape(-1)]
  

def load_embeddings(name="tr-word2vec-model_v3.bin"):
  """Loads a pretrained word embedding model into memory."""
  word2vec_bin = os.path.join(_EMBEDDING_DIR, name)
  word2vec = gensim.models.Word2Vec.load(word2vec_bin)
  
  return word2vec


def maxlen(data):
  """Returns the length of the longest list in a list of lists."""
  return len(max(data, key=len))


def annotate_bio_spans(sentence):
  """Annotatest BIO spans in a sentence.
  
  This function gets a sentence where semantic role labels are only
  marked on the immediate child tokens of the verb and propagates these
  labels using the BIO tagging framework to all the children of those
  tokens. 
  
  Example:
    input = The boy_AO loves the dog_A1 very much.
    output = The_B-AO boy_I-AO loves the_B-A1 cat_B-A1 very_O much_O.
  
  Args:
    sentence: a sentence_pb2.Sentence object.
  Returns:
    sentence: a sentence_pb2.Sentence object.
  
  """
  heads = defaultdict(list)
  srls = Counter()
  for token in sentence.token:
    if token.index == 0: 
      continue
    heads[token.selected_head.address].append(token)
  heads = OrderedDict(sorted(heads.items(), reverse=True))
  
  for head in heads.keys():
    if not sentence.token[head].HasField("srl"):
      continue
    children = heads[head]
    for child in children:
      child.srl = sentence.token[head].srl

  for token in sentence.token:
    if token.index == 0:
      continue
    if not token.HasField("srl"):
      token.srl = "O"
    else:
      if srls[token.srl] == 0:
        srls[token.srl] += 1
        token.srl = "B-" + token.srl
      else:
        token.srl = "I-" + token.srl    

  return sentence
  

def get_argument_spans(sentence, token_index, predicate_index, argument_span=[]):
  """Returns all the argument span of an argument based on the head.
  
  Args:
    sentence_pb2.Sentence()
    token_index = the index of the token in the sentence.
  Returns:
    span_indices: set of token indices which represent the argument span.
  """
  # Special treatment for "case" tokens because of the weird way
  # they are annotated in Turkish propbank.
  if sentence.token[token_index].label == "case":
    head_address = sentence.token[token_index].selected_head.address
    head_of_span = sentence.token[head_address].index
    if not sentence.token[head_of_span] in argument_span:
      argument_span.append(sentence.token[head_of_span])
  else:
    head_of_span = token_index

  for token in sentence.token:
    if token in argument_span:
      continue
    if token.index == predicate_index:
      continue
    if token.selected_head.address == head_of_span:
      argument_span.append(token)
      get_argument_spans(sentence, token.index, predicate_index, argument_span)
  span_indices = set([token.index for token in argument_span])
  return list(span_indices)

if __name__ == "__main__":
  trb = reader.ReadTreebankTextProto(
    "data/testdata/propbank/propbank_ud_testdata_proto.pbtxt"
  )
  sentence = trb.sentence[1]
  annotate_bio_spans(sentence)
