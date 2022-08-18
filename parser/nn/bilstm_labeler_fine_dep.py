
import tensorflow as tf
import numpy as np


from sklearn.metrics import classification_report
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers
from parser.nn import load_models
from tagset.reader import LabelReader
Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset


class Labeler:
  def __init__(self, word_embeddings, n_units=256, n_output_classes=43, use_pos=True, use_morph=True,
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
    concat = tf.keras.layers.Concatenate(name="concat")([w_embedding, s_w_embedding, pos_embedding, morph_input])
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
  dev_treebank = "tr_boun-ud-dev.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
  train_dataset, dev_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                                   train_treebank=train_treebank,
                                                                   batch_size=250,
                                                                   dev_treebank=dev_treebank,
                                                                   dev_batch_size=25,
                                                                   test_treebank=test_treebank,
                                                                   test_batch_size=1000,
                                                                   type="pbtxt")
  # class_weights = {7: 1.0, 3: 1.0, 0: 1.0, 4: 3.0,s
  #                  1: 1.96, 2: 1.88, 5: 3.65, 6: 2.78}
  class_weights = {0: 0.01, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,
                   12: 3.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 3.0, 20: 1.0, 21: 1.0, 22: 1.0,
                   23: 0.01, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 0.01, 29: 1.0, 30: 1.0, 31: 1.0, 32: 3.0, 33: 1.0,
                   34: 3.0, 35: 1.0, 36: 1.0, 37: 1.0, 38: 0.01, 39: 4.0, 40: 1.0, 41: 3.0, 42: 0.01}
  class_names = list(LabelReader.get_labels("dep_labels").labels.keys())
  # move pad to the beginning
  class_names.remove("-pad-")
  class_names = ["pad"] + class_names
  print("class names ", class_names, len(class_names))
  model = Labeler(word_embeddings=prep.word_embeddings,
                  n_output_classes=label_feature.n_values).build_model(show_summary=True)
  step = 0

  for epoch in range(10):
    loss_tracker = []
    acc_tracker = []
    val_loss_tracker = []
    val_acc_tracker = []
    for data, dev_data in zip(train_dataset, dev_dataset):
      step += 1
      print("batch ", step)
      words, pos, morph, dep_labels = data["words"], data["pos"], data["morph"], data["dep_labels"]
      # print("dep labels ", dep_labels)
      # converted_dep_labels = label_converter.convert_labels(dep_labels)
      # print(converted_dep_labels)
      # input()

      dev_w, dev_p, dev_m, dev_dep = dev_data["words"], dev_data["pos"], dev_data["morph"], \
                                     dev_data["dep_labels"]
      # converted_test_labels = label_converter.convert_labels(test_dep)
      # print("test dep labels ", test_dep)
      # print("converted test labels ", converted_test_labels)
      # input()
      history = model.fit([words, pos, morph], dep_labels,
                          validation_data=([dev_w, dev_p, dev_m], dev_dep),
                          class_weight=class_weights,
                          epochs=1)
      # print(history.history.keys())
      loss_tracker.append(history.history["loss"][0])
      acc_tracker.append(history.history["accuracy"][0])
      val_loss_tracker.append(history.history["val_loss"][0])
      val_acc_tracker.append(history.history["val_accuracy"][0])
      # print(loss_tracker, acc_tracker, val_loss_tracker, val_acc_tracker)
      # input()
    print(f"end of epoch {epoch}")
    print("--------------------> metrics <-------------------------")
    print("epoch mean loss: ", np.mean(loss_tracker))
    print("epoch mean acc: ", np.mean(acc_tracker))
    print("epoch mean val loss: ", np.mean(val_loss_tracker))
    print("epoch mean val acc: ", np.mean(val_acc_tracker))

    # see class metrics on test set
    test_data = test_dataset.get_single_element()
    test_ex_w, test_ex_p, test_ex_m, test_ex_dep = (test_data["words"], test_data["pos"],
                                                    test_data["morph"], test_data["dep_labels"])
    # print("test ex dep ", test_ex_dep)
    # input()
    test_ex_dep_r = tf.squeeze(tf.reshape(test_ex_dep, (1, test_ex_dep.shape[0]*test_ex_dep.shape[1])))
    # print("test ex dep r ", test_ex_dep_r)
    scores = model([test_ex_w, test_ex_p, test_ex_m])
    preds = tf.argmax(scores, axis=2)
    # print("preds ", preds)
    # input()
    preds_r = tf.squeeze(tf.reshape(preds, (1, preds.shape[0]*preds.shape[1])))
    # print("preds r ", preds_r)
    # input()
    report = classification_report(y_true=test_ex_dep_r, y_pred=preds_r, output_dict=True)
    for k, v in report.items():
      print(k, v)
      print("\n")
    c = input("Start new epoch: y/n")
    if c == "n":
      break



  # If you want to inspect some results manually at the end of training.
  for step, test_example in enumerate(test_dataset):
    # print(test_example)
    # input()
    test_ex_w, test_ex_p, test_ex_m, test_ex_dep = (test_example["words"], test_example["pos"],
                                                    test_example["morph"], test_example["dep_labels"])
    # converted_test_ex_labels = label_converter.convert_labels(test_ex_dep)
    print("true labels ", test_ex_dep)
    input()
    scores = model([test_ex_w, test_ex_p, test_ex_m])
    print("pred scores ", scores)
    print("predicted labels ", tf.argmax(scores, axis=2))
    input()
    if step > 15:
      break


