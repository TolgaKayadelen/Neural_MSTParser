import os
import tensorflow as tf


from parser.nn import base_parser, architectures
from util.nn import nn_utils
from util import converter
from input import embeddor, preprocessor
from proto import metrics_pb2
from tensorflow.keras import layers, metrics, losses, optimizers


class LabelFirstParser(base_parser.BaseParser):
  """A label first parser predicts labels before predicting heads."""
  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)


  def _training_metrics(self):
    return {
      "heads": metrics.SparseCategoricalAccuracy(),
      "labels": metrics.SparseCategoricalAccuracy()
    }

  @property
  def _head_loss_function(self):
    """Returns loss per token for head prediction.
    As we use the SparseCategoricalCrossentropy function, we expect the target labels
    to be provided as integers indexing the correct labels rather than one hot vectors.
    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    """
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)
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
    label_inputs = tf.keras.Input(shape=(None, ), name="labels")
    input_dict = {"words": word_inputs}
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    if self._use_dep_labels:
      input_dict["labels"] = label_inputs
    return input_dict

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}, dep_labels: {self._use_dep_labels}""")
    model = architectures.LabelFirstParsingModel(
      n_dep_labels=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      predict=self._predict,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      use_dep_labels=self._use_dep_labels
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

  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["heads", "labels"],
                            features=["words", "pos", "morph"],
                            model_name="tests_base_parser")

  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="treebank_train_0_50.pbtxt"
  test_treebank = "treebank_test_0_10.conllu"
  train_sentences = prep.prepare_sentence_protos(path=os.path.join(_DATA_DIR, train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=10
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=1
    )
  else:
    test_dataset=None
  metrics = parser.train(dataset=dataset, epochs=10, test_data=test_dataset)
  print(metrics)
