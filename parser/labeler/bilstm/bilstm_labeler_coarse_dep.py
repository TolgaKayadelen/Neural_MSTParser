
import tensorflow as tf
import numpy as np


from typing import Dict, Tuple, List
from parser.nn import base_parser
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers
from parser.nn import load_models
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util.nn import label_converter
from util import writer

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset


class Labeler:
  def __init__(self, word_embeddings, n_units=256, n_output_classes=8, use_pos=True, use_morph=True,
               name="LSTM_Labeler_Wrapper" ):
    self.word_embeddings = word_embeddings
    self.n_units=n_units
    self.n_output_classes=n_output_classes
    self.use_pos = use_pos
    self.use_morph=use_morph

  def build_model(self, show_summary=True):
    word_input = tf.keras.Input(shape=(None,), name="words")
    pos_input = tf.keras.Input(shape=(None,), name="pos")
    morph_input = tf.keras.Input(shape=(None, 56), name="morph")
    w_embedding = tf.keras.layers.Embedding(input_dim=self.word_embeddings.vocab_size,
                                            output_dim=self.word_embeddings.embedding_dim,
                                            weights=[self.word_embeddings.index_to_vector],
                                            trainable=False,
                                            name="pretrained_word_embeddings")(word_input)
    s_w_embedding = tf.keras.layers.Embedding(input_dim=self.word_embeddings.vocab_size,
                                              output_dim=300,
                                              trainable=True,
                                              name="learned_word_embeddings")(word_input)
    pos_embedding = tf.keras.layers.Embedding(input_dim=37, output_dim=32,
                                              name="pos_embeddings",
                                              trainable=True)(pos_input)
    concat = tf.keras.layers.Concatenate(name="concat")(
      [w_embedding, s_w_embedding, pos_embedding, morph_input])
    lstm1 = layers.Bidirectional(layers.LSTM(
      units=self.n_units, return_sequences=True, name="lstm1"))(concat)
    dropout1 = layers.Dropout(rate=0.3, name="dropout1")(lstm1)
    lstm2 = layers.Bidirectional(layers.LSTM(
      units=128, return_sequences=True, name="lstm2"))(dropout1)
    dropout2 = layers.Dropout(rate=0.3, name="dropout2")(lstm2)
    lstm3 = layers.Bidirectional(layers.LSTM(
      units=64, return_sequences=True, name="lstm2"))(dropout2)
    model_output = layers.Dense(self.n_output_classes, activation='softmax',
                                kernel_initializer='glorot_uniform',
                                name=f'labels')(lstm3)
    model = tf.keras.Model([word_input, pos_input, morph_input], model_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == "__main__":
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)
  train_treebank= "tr_boun-ud-train.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=8,
                                                      test_treebank=test_treebank,
                                                      test_batch_size=1,
                                                      type="pbtxt")
  class_weights = {7: 1.0, 3: 1.0, 0: 1.0, 4: 3.0,
                   1: 1.96, 2: 1.88, 5: 3.65, 6: 2.78}
  # _metrics = parser.train(dataset=train_dataset, epochs=200, test_data=test_dataset)
  model = Labeler(word_embeddings=prep.word_embeddings).build_model(show_summary=True)
  step = 0

  for epoch in range(10):
    loss_tracker = []
    acc_tracker = []
    val_loss_tracker = []
    val_acc_tracker = []
    for data, test_data in zip(train_dataset, test_dataset):
      step += 1
      print("batch ", step)
      # test_data = test_dataset.get_single_element()
      # print(test_data["tokens"].shape)
      # print(test_data["dep_labels"].shape)
      # input()
      words, pos, morph, dep_labels = data["words"], data["pos"], data["morph"], data["dep_labels"]
      converted_dep_labels = label_converter.convert_labels(dep_labels)
      # print(converted_dep_labels)
      # input()

      test_w, test_p, test_m, test_dep = test_data["words"], test_data["pos"], test_data["morph"], \
                                         test_data["dep_labels"]
      converted_test_labels = label_converter.convert_labels(test_dep)
      # print("converted test labels ", converted_test_labels)
      # input()
      history = model.fit([words, pos, morph], converted_dep_labels,
                          validation_data=([test_w, test_p, test_m], converted_test_labels),
                          class_weight=class_weights,
                          epochs=2)
      # print(history.history.keys())
      loss_tracker.append(history.history["loss"][1])
      acc_tracker.append(history.history["accuracy"][1])
      val_loss_tracker.append(history.history["val_loss"][1])
      val_acc_tracker.append(history.history["val_accuracy"][1])
      # print(loss_tracker, acc_tracker, val_loss_tracker, val_acc_tracker)
      # input()
    val_acc = np.mean(val_acc_tracker)
    acc = np.mean(acc_tracker)
    print(f"end of epoch {epoch}")
    print("--------------------> metrics <-------------------------")
    print("epoch mean loss: ", np.mean(loss_tracker))
    print("epoch mean acc: ", acc)
    print("epoch mean val loss: ", np.mean(val_loss_tracker))
    print("epoch mean val acc: ", val_acc)
    c = input("Start new epoch: y/n")
    if acc > 0.98 and val_acc < 0.85:
      c = input("continue training?")
      if  c == "n":
        break
    if acc > 0.85:
      c = input("continue training?")
      if  c == "n":
        break
  for step, test_example in enumerate(test_dataset):
    print(test_example)
    input()
    test_ex_w, test_ex_p, test_ex_m, test_ex_dep = (test_example["words"], test_example["pos"],
                                                    test_example["morph"], test_example["dep_labels"])
    converted_test_ex_labels = label_converter.convert_labels(test_ex_dep)
    print("true labels ", converted_test_ex_labels)
    preds = model([test_ex_w, test_ex_p, test_ex_m])
    print("preds ", preds)
    print("predicted labels ", tf.argmax(preds, axis=2))
    input()
    if step > 15:
      break

  # print(output)
  # print(_metrics)
  # writer.write_proto_as_text(metrics,
  #                            f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  # parser.save_weights()
  # logging.info(f"{parser.model_name} results written")

