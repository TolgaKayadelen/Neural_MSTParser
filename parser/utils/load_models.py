# This module loads pretrained models and makes sure that relevant
# weights are shared between them.

import os
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.dep.lfp import label_first_parser
from parser.utils import layer_utils
# from parser.nn import bilstm_labeler

def load_word_embeddings():
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  return word_embeddings


def load_preprocessor(*, word_embeddings=None, head_padding_value=0, one_hot_features=[], language="tr"):
  if language == "tr":
    prep = preprocessor.Preprocessor(
      word_embeddings=word_embeddings,
      features=["words", "pos", "morph", "heads", "category", "dep_labels", "sent_id"],
      labels=["heads"],
      head_padding_value=head_padding_value,
      one_hot_features=one_hot_features,
      language=language,
    )
  elif language == "en":
    prep = preprocessor.Preprocessor(
      features=[ "words", "pos", "heads", "category", "dep_labels", "sent_id"],
      labels=["heads"],
      head_padding_value=head_padding_value,
      one_hot_features=one_hot_features,
      language=language,
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


def load_data(*, preprocessor: preprocessor.Preprocessor,
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
