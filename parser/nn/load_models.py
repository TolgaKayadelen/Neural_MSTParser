# This module loads pretrained models and makes sure that relevant
# weights are shared between them.

import os
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn import label_first_parser
from parser.nn import layer_utils
from parser.nn import bilstm_labeler

def load_word_embeddings():
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  return word_embeddings


def load_preprocessor(word_embeddings, head_padding_value=0):
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"],
    head_padding_value=head_padding_value
  )
  return prep


def load_labeler(labeler_name, prep, path=None):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  labeler = bilstm_labeler.BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                                         n_output_classes=label_feature.n_values,
                                         predict=["labels"],
                                         features=["words", "pos", "morph"],
                                         model_name="dependency_labeler")
  labeler.load_weights(name=labeler_name, path=path)
  return labeler, label_feature


def load_parser(parser_name, prep, path=None):
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  parser = label_first_parser.LabelFirstParser(word_embeddings=prep.word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=["heads"],
                                               features=["words", "pos", "morph", "heads", "dep_labels"],
                                               model_name="label_first_parser")
  parser.load_weights(name=parser_name, path=path)
  return parser


def load_data(preprocessor: preprocessor.Preprocessor,
              train_treebank: str,
              batch_size: int,
              test_treebank: str = None):
  _DATA_DIR = "data/UDv29/train/tr"
  _TEST_DATA_DIR = "data/UDv29/dev/tr"
  train_sentences = preprocessor.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank)
  )
  train_dataset = preprocessor.make_dataset_from_generator(
    sentences=train_sentences, batch_size=batch_size
  )

  if test_treebank is not None:
    test_sentences = preprocessor.prepare_sentence_protos(
      path=os.path.join(_TEST_DATA_DIR, test_treebank)
    )
    test_dataset = preprocessor.make_dataset_from_generator(
      sentences = test_sentences,
      batch_size=1
    )
  else:
    test_dataset = None

  return train_dataset, test_dataset
