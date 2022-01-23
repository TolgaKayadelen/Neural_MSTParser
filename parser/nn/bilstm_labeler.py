import os
import logging
import tensorflow as tf

from parser.nn import base_parser, architectures
from util.nn import nn_utils
from util import converter, writer
from input import embeddor, preprocessor
from proto import metrics_pb2
from tensorflow.keras import layers, metrics, losses, optimizers


class BiLSTMLabeler(base_parser.BaseParser):
  """A bi-lstm labeler that can be used for any kind of sequence labeling tasks."""
  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)


  def _training_metrics(self):
    return {
      "labels": metrics.SparseCategoricalAccuracy()
    }

  @property
  def _head_loss_function(self):
    """No head loss function for labeler."""
    return None

  @property
  def _label_loss_function(self):
    """Returns loss per token for label prediction.

    As we use the SparseCategoricalCrossentropy function, we expect the target labels to be
    to be provided as integers indexing the correct labels rather than one hot vectors. The predictions
    should be keeping the probs as float values for each label per token.

    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy"""

    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None, ), name="words")
    pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    input_dict = {"words": word_inputs}
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    return input_dict

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}""")
    model = architectures.LSTMLabelingModel(
      n_output_classes=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
    )
    model(inputs=self.inputs)
    return model


if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                         n_output_classes=label_feature.n_values,
                         predict=["labels"],
                         features=["words", "pos", "morph"],
                         model_name="dependency_labeler")
  # parser.load_weights(name="tests_saving")
  # for w in parser.model.weights:
  #  print(type(w))
  # print(parser.model.weights[-2])
  # weights = parser.model.get_weights()
  # for layer in parser.model.layers:
  #  print(layer.name)
  #  if layer.name == "labels":
  #    print("working the labels layer")
  #    trainable_weights = layer.trainable_weights
  #    print("trainable weights ", trainable_weights)
  #    print(tf.math.reduce_sum(trainable_weights[0], axis=0))
  print("parser ", parser)
  # input("press to cont.")

  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="tr_imst_ud_train_dev.pbtxt"
  test_treebank = "tr_imst_ud_test_fixed.pbtxt"
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=250
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(
      path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=1)
  else:
    test_dataset=None
  metrics = parser.train(dataset=dataset, epochs=1, test_data=test_dataset)
  # metrics = parser.test(dataset=test_dataset)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")
