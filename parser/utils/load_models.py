# This module loads pretrained models and makes sure that relevant
# weights are shared between them.

import os
import json
import pickle
import numpy as np
import sys
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.dep.lfp import label_first_parser
from parser.utils import layer_utils
from parser.labeler.bilstm import bilstm_labeler

def load_word_embeddings():
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  return word_embeddings


def load_preprocessor(*, word_embedding_indexes,
                      language,
                      features=["words", "pos"], # word, pos, morph
                      pos_indexes=None,
                      head_padding_value=0,
                      one_hot_features=[],
                      embedding_type="word2vec"):
  allow_list = ["words", "pos", "morph", "category", "dep_labels"]
  print("features ", features)
  # input()
  assert set(features).issubset(allow_list), "Can use words, pos, morph, category, dep_labels as features!"
  prep = preprocessor.Preprocessor(
    word_embedding_indexes=word_embedding_indexes,
    pos_indexes=pos_indexes,
    features=features,
    labels=["heads"],
    head_padding_value=head_padding_value,
    one_hot_features=one_hot_features,
    language=language,
    embedding_type=embedding_type
  )
  return prep


def load_labeler(labeler_name, prep, k=5):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  labeler = bilstm_labeler.BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                                         n_output_classes=label_feature.n_values,
                                         predict=["labels"],
                                         features=["words", "pos", "morph"],
                                         model_name=labeler_name,
                                         top_k=True,
                                         k=k,
                                         test_every=0)
  labeler.load_weights(name=labeler_name)
  print("labeler ", labeler)
  return labeler

"""
# The predict and features lists are configured to load "label_first_parser_gold_morph_labels"
def load_parser(parser_name, prep, test_every=0, one_hot_labels=False,
                predict=["heads"], features=["words", "morph", "dep_labels"]):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  parser = label_first_parser.LabelFirstParser(word_embeddings=prep.word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=predict,
                                               features=features,
                                               test_every=test_every,
                                               model_name=parser_name,
                                               one_hot_labels=one_hot_labels)
  parser.load_weights(name=parser_name)
  print("parser ", parser)
  return parser
"""

# The predict and features lists are configured to load "label_first_predicted_head_gold_labels_only"
def load_parser(*, parser_name, prep, word_embeddings, test_every=0, one_hot_labels=False,
                predict=["heads"], features=["words", "dep_labels"], language="tr"):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  parser = label_first_parser.LabelFirstParser(language=language,
                                               word_embeddings=word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=predict,
                                               features=features,
                                               test_every=test_every,
                                               model_name=parser_name,
                                               one_hot_labels=one_hot_labels)
  parser.load_weights(name=parser_name)
  print("parser ", parser)
  return parser

# This loads the parser named tr_lfp_gold_pos_and_labels_20230131_040723
def load_pos_gold_parser(parser_name, prep, word_embeddings, test_every=0, one_hot_labels=False,
                         predict=["heads"], features=["words", "pos", "dep_labels"]):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  parser = label_first_parser.LabelFirstParser(word_embeddings=word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=predict,
                                               features=features,
                                               test_every=test_every,
                                               model_name=parser_name,
                                               one_hot_labels=one_hot_labels)
  parser.load_weights(name=parser_name)
  print("parser ", parser)
  return parser

def load_category_gold_parser(parser_name, prep, word_embeddings, test_every=0, one_hot_labels=False,
                              predict=["heads"], features=["words", "category", "dep_labels"]):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  parser = label_first_parser.LabelFirstParser(word_embeddings=word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=predict,
                                               features=features,
                                               test_every=test_every,
                                               model_name=parser_name,
                                               one_hot_labels=one_hot_labels)
  parser.load_weights(name=parser_name)
  print("parser ", parser)
  return parser


def load_data(*,
              preprocessor: preprocessor.Preprocessor,
              train_treebank: str = None,
              dev_treebank: str = None,
              test_treebank: str = None,
              data_dir: str = None,
              dev_data_dir: str = None,
              test_data_dir: str = None,
              type="pbtxt",
              batch_size: int = None,
              dev_batch_size: int = None,
              test_batch_size=1,
              language="tr",
              ):
  if data_dir is not None:
    data_dir = data_dir
  else:
    data_dir = "data/UDv29/train/" + language
  if dev_data_dir is not None:
    dev_data_dir = dev_data_dir
  else:
    dev_data_dir = "data/UDv29/dev/" + language
  if test_data_dir is not None:
    test_data_dir = test_data_dir
  else:
    test_data_dir = "data/UDv29/test/" + language
  if type=="pbtxt":
    if train_treebank is not None:
      train_sentences = preprocessor.prepare_sentence_protos(
        path=os.path.join(data_dir, train_treebank)
      )
      train_dataset = preprocessor.make_dataset_from_generator(
        sentences=train_sentences, batch_size=batch_size
      )
    else:
      train_dataset = None
    if dev_treebank is not None:
      dev_sentences = preprocessor.prepare_sentence_protos(
        path=os.path.join(dev_data_dir, dev_treebank)
      )
      dev_dataset = preprocessor.make_dataset_from_generator(
        sentences = dev_sentences,
        batch_size=dev_batch_size
      )
    else:
      dev_dataset = None

    if test_treebank is not None:
      test_sentences = preprocessor.prepare_sentence_protos(
        path=os.path.join(test_data_dir, test_treebank)
     )
      test_dataset = preprocessor.make_dataset_from_generator(
        sentences = test_sentences,
        batch_size=test_batch_size
      )
    else:
      test_dataset = None
  elif type =="tfrecords":
    if train_treebank is not None:
      train_dataset = preprocessor.read_dataset_from_tfrecords(
        records=os.path.join(data_dir, train_treebank),
        batch_size=batch_size
      )
    if dev_treebank is not None:
      dev_dataset = preprocessor.read_dataset_from_tfrecords(
        records=os.path.join(dev_data_dir, dev_treebank),
        batch_size=dev_batch_size
      )
    else:
      dev_dataset = None
    if test_treebank is not None:
      test_dataset = preprocessor.read_dataset_from_tfrecords(
        records=os.path.join(test_data_dir, test_treebank),
        batch_size=test_batch_size
      )
    else:
      test_dataset = None

  else:
    raise ValueError("Invalid data type requested.")

  return train_dataset, dev_dataset, test_dataset

def load_layer_weights(weights_file):
  """Loads a pretrained layer weights file (extension .npy) from weights dir."""
  weights_dir = "./parser/layer_weights"
  if not weights_file.endswith(".npy"):
    weights_file += ".npy"
  return np.load(os.path.join(weights_dir, weights_file))


def load_i18n_embeddings(language="zh"):
  file = f"./embeddings/{language}/{language}.vectors"
  conll_token_to_index_dict = {"-pad-": 0, "-oov-": 1}
  embed_matrix = {}
  counter = 2
  with open(file, "rb") as f:
    for linenum, line in enumerate(f):
      if linenum == 0:
        continue
      if linenum % 100000 == 0:
        print(linenum)
      line = line.strip().split()
      if line:
        embed_matrix[line[0]] = np.array(line[1:], dtype=np.float32)
        conll_token_to_index_dict[line[0]] = counter
        counter += 1
    word_embeddings = embeddor.Embeddings(name=f"{language}_embeddings",
                                          matrix=embed_matrix,
                                          type="fasttext")
    # print(word_embeddings.token_to_index)
    # print(word_embeddings.vocab)
    print(word_embeddings.sanity_check)
    assert (word_embeddings.sanity_check != embeddor.SanityCheck.FAIL)
    assert (word_embeddings.vocab[0] == "-pad-" and word_embeddings.vocab[1] == "-oov-")
    # pickle_save_file = f"./input/{language}_conll_token_to_index_dict.pkl"
    # with open(pickle_save_file, 'wb') as handle:
    #   pickle.dump(conll_token_to_index_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return word_embeddings


def write_conll_token_to_index_dict(language="tr"):
  file = f"./embeddings/{language}/{language}.vectors"
  conll_token_to_index_dict = {"-pad-": 0, "-oov-": 1}
  counter = 2
  with open(file, "rb") as f:
    for linenum, line in enumerate(f):
      if linenum == 0:
        continue
      if linenum % 100000 == 0:
        print(linenum)
      line = line.strip().split()
      if line:
        # print("token ", line[0])
        conll_token_to_index_dict[line[0]] = counter
        counter += 1
  pickle_save_file = f"./input/{language}_conll_token_to_index_dict.pkl"
  with open(pickle_save_file, 'wb') as handle:
    pickle.dump(conll_token_to_index_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_embedding_indexes():
  with open('./transformer/hf/dataset/tr/token_to_index_dictionary.pkl', 'rb') as f:
    loaded =  pickle.load(f)
  return loaded

if __name__ == "__main__":
  load_i18n_embeddings()
  # write_conll_token_to_index_dict()