"""A BiLSTM sequence model."""

import os
import tensorflow as tf
import numpy as np
import gensim
import argparse

from util import writer
from util.nn import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class BiLSTM:
  """A BiLSTM model for sequence labeling."""
  def __init__(self):
    self.model = None
    
  def pretrained_embeddings_layer(self, word2vec, words_to_index, embedding_dim):
    """Creates an embedding layer for the Neural Net and feeds a pretrained word2vec model into it.
    
    Args:
      word2vec: a pretrained word embeddings model.
      word_to_index: dictionary where keys are words and values are indices.
      embedding_dim: dimension of the embedding vector.
    Returns:
      embedding_layer: A pretrained Keras Embedding() layer with weights set as word2vec weights.
    """
    # word2vec_bin = 'tr-word2vec-model_v3.bin' 
    #word2vec = gensim.models.Word2Vec.load(word2vec_bin)
    
    print(words_to_index["tolga"]) # should return 493047
    
    # sanity checking
    vocab_len = len(words_to_index)
    assert(len(word2vec.wv.vocab.keys()) == vocab_len-2)
    
    # get the dimension of the embedding vector
    # embedding_dim = word2vec[index_to_words[100]].shape[0]
    
    # Initialize the embedding matrix as an array of zeros
    embedding_matrix = np.zeros(shape=(vocab_len, embedding_dim))
    
    # Set each row index of the embedding matrix to be the word vector of the
    # word in that index in the vocabulary.
    for word, idx in words_to_index.items():
      if word == "-pad-":
        embedding_matrix[idx, :] = np.zeros(shape=(embedding_dim,))
      elif word == "-oov-":
        # random initialization for oov items.
        embedding_matrix[idx, :] = np.random.randn(embedding_dim,)
      else:
        embedding_matrix[idx, :] = word2vec[word]
    
    logging.info(f"Shape of the embedding matrix: {embedding_matrix.shape}")
    
    # Initialize the Keras Embedding Layer. This will return an embedding layer
    # of shape (None, None, 300). The first None is preserved for batch size,
    # and the second None is preserved for the sequence_length. For example, if
    # you give this layer an input of shape (32, 10), i.e. a (32,10) matrix
    # where 32 is number of examples and 10 is the length of each example, you
    # will get as output a 3-dimensional array of (32, 10, 300) where 300 is
    # the dimension of the embedding vector for each word in the (32,10) array.
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_len,
                                                output_dim=embedding_dim,
                                                trainable=False)
    
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer
    
  def index_words(self, embedding_model):
    """Indexes all the vocabulary items in the embedding model.
    
    Args:
      embedding_model: a pretrained embedding model.
    Returns:
      words_to_index: dict, pairs of words as keys and indices as values.
      index_to_word: dict, reverse of the word_to_index dict.
    """
    words_to_index = {}
    index_to_words = {}
    words = set(embedding_model.wv.vocab.keys())
    logging.info(f"total number of words: {len(words)}")
    
    # Preserve index 0 for padding key.
    words_to_index["-pad-"] = 0
    index_to_words[0] = "-pad-"
    # Preserve index 1 for oov.
    words_to_index["-oov-"] = 1
    index_to_words[1] = "-oov-"
    
    i = 2
    for w in sorted(words):
      words_to_index[w] = i
      index_to_words[i] = w
      i += 1
    return words_to_index, index_to_words
  
  def index_sentences(self, sentences, word_indices, maxlen):
    """Indexes all the words in a set of sentences. 
    
    The indexing is made based on the word_indices dictioanary that is obtained from the
    word embedding model. 
    
    The output shape should be such that it can given to an Embedding() layer. 
    
    Args:
      sentences: list of lists, where each list is a list of words representing a sentence.
      word_indices: dictionary containing each word mapped to an index.
      maxlen: scalar, maximum number of words in a sentence.
    
    Returns:
      sentence_indices: array of indices corresponding of each word in each sentence
        of shape (m, maxlen)
    
    Sentences which have shorter than maxlen words are padded with 0s on the right.
    """
    m = len(sentences)
    sentence_indices = np.zeros(shape=(m, maxlen))
    for i in range(len(sentences)):
      sentence = sentences[i]
      j = 0
      
      for w in sentence:
        try:
          sentence_indices[i, j] = word_indices[w]
        except KeyError:
          sentence_indices[i, j] = word_indices["-oov-"]
        j += 1
    
    return sentence_indices
    

  def index_labels(self, labels, label_dict, n_sentences, maxlen=None, pad=True):
    """Gets a sequence of labels and converts them into a sequence of indices."""
    
    def _labels_to_index(labels):
      labels_to_index = []
      for label in labels:
        labels_to_index.append(label_dict[label])
      indexed = np.asarray(labels_to_index)
  
      if pad and maxlen:
        pad_value = maxlen - len(indexed)
        padded = np.pad(indexed, (0, pad_value), "constant", constant_values=(label_dict["-pad-"]))
        return padded
      return indexed
  
    label_indices = np.zeros(shape=(n_sentences, maxlen, len(label_dict)))
    for i, label_set in enumerate(labels):
      indexed = _labels_to_index(label_set)
      # print(i)
      # print(indexed)
      one_hot = nn_utils.convert_to_one_hot(indexed, len(label_dict))
      # print(one_hot)
      label_indices[i, :, :] = one_hot
    return label_indices
    

  def _create_model(self, input_shape, word2vec, word_indices, n_classes,
                    additional_input=None, embedding_dim=None):
    """Creates a BiLSTM model.
    
    Args:
      input_shape: shape of the input, usually (max_len, )
      word2vec: the word embeddings model.
      word_indices: dict, key:value pairs of words and indices.
      n_classes: number of classes in the label set.
      additional_input: a dictionary of values. See .train() for details.
    
    Returns:
      a Keras Model instance.
    """
    
    # Define sentence_indices as the Input to the graph. It has dtype=int32
    # because it contains indices for each word in the sentence, which are
    # integers. This will be of shape (batch_size, maxlen)
    sentence_indices = tf.keras.layers.Input(shape=input_shape, dtype="int32", name="sentences")
    print(f"sentence input shape: {sentence_indices.shape}")

    if additional_input:
      additional = tf.keras.layers.Input(shape=(additional_input["shape"]),
                                         dtype="float32", 
                                         name=additional_input["name"])
      print(f"{additional_input['name']} input shape: {additional.shape}")
    # Create the embedding layer with the word2vec model.
    embedding_layer = self.pretrained_embeddings_layer(word2vec,
                                                       word_indices, 
                                                       embedding_dim)
    
    # Propagate sentences through the embedding layer.
    embeddings = embedding_layer(sentence_indices)
    
    # If there's additional input, concatenate it with word embedding output.
    if additional_input:
      X = tf.keras.layers.Concatenate()([embeddings, additional])      
    else:
      X = embeddings

    # Propagate the embeddings through an LSTM layer with 128 hidden units.
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
      units=128, return_sequences=True))(X)
    
    # Add dropout.
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    
    # Propagate through another LSTM layer.
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
      units=128, return_sequences=True))(X)
    
    # Add dropout.
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    
    # Add the Dense Layer.
    X = tf.keras.layers.Dense(units=n_classes)(X)
    # Add activation
    X = tf.keras.layers.Activation("softmax")(X)
    
    # Create the model.
    if additional_input:
      model = tf.keras.Model(inputs=[sentence_indices, additional], outputs=X)
    else:
      model = tf.keras.Model(inputs=sentence_indices, outputs=X)
    
    return model


  def train(self, train_data, train_data_labels, label_dict, epochs, embeddings=False,
            loss="categorical_crossentropy", optimizer="adam", batch_size=None,
            vld_data=None, vld_data_labels=None, test_data=None, test_data_labels=None,
            additional_input=None, additional_input_vld=None, additional_input_test=None):
      """Trains the LSTM model.
      
      Args:
        train_data: list of lists. Each list is a sequence of words representing one sentence.
        train_data_labels: list of lists. Each list is a sequence of tags.
        label_dict: dictionary of labels where each label is mapped to an integer value. 
        epochs: int, number of epochs. 
        embeddings: bool, whether to use pretrained embeddings.
        loss: the cost function. 
        optimizer: the optimization algorithm to use.
        vld_data: list of lists. Validation data.
        vld_data_labels: list of lists. Validation data labels.
        batch_size: the batch size use for dividing training data into batches.
        test_data: list of sentences. The data to test the model on. 
        test_data_labels: list of lists. Similar to train_labels.
        additonal_input: dictionary with following values:
           name: name of the additional inputs.
           list of lists: each list represents the sequence, where each value 
           in each list logs a value for each token (e.g. its postag index). 
      """
      # ---- PREPARE EMBEDDING LAYER ----
      if embeddings: 
        word2vec = nn_utils.load_embeddings()
        words_to_index, index_to_words = self.index_words(word2vec)
        embedding_dim = word2vec[index_to_words[100]].shape[0]
      
      # ---- GET MAXLEN SEQUENCE ------
      maxlen = nn_utils.maxlen(train_data)
      logging.info(f"maxlen: {maxlen}")
      
      # ---- PREPARE TRAINING DATA -----
      # Index the words in the sentences based on the word2ved indices. This 
      # returns a numpy array of (m, maxlen), which is then the main input
      # to the model.
      train_sentences = self.index_sentences(train_data, words_to_index, maxlen)
      
      # ---- PREPARE THE LABELS --------
      self.label_dict = label_dict
      self.reverse_label_dict = {v:k for k,v in label_dict.items()}
      
      # Indexes the training labels. This returns a 3-D array of (m, maxlen, n_labels)
      # maxlen is the sequence length, and n_labels is the total number of labels.
      train_labels = self.index_labels(train_data_labels, label_dict, len(train_data), maxlen)
      logging.info(f"shape of the output {train_labels.shape}")
      
      
      # ----- SET UP THE MODEL -------
      self.model = self._create_model(input_shape=(maxlen,),
                                      word2vec=word2vec,
                                      word_indices=words_to_index,
                                      additional_input=additional_input,
                                      n_classes=len(label_dict),
                                      embedding_dim=embedding_dim
                                      )
      # Print model summary.
      print(self.model.summary())
      
      # TODO: install pydot and plot the model.
      # tf.keras.utils.plot_model(model, "my_model.png", show_shapes=True)
      
      # Compile the model.
      self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
      
      
      # ------ TRAIN THE MODEL ------
      if vld_data and vld_data_labels:
        vld_sentences = self.index_sentences(vld_data, words_to_index, maxlen)
        vld_labels=self.index_labels(vld_data_labels, label_dict, len(vld_data), maxlen)
        if additional_input_vld:
          self.model.fit(
            {"sentences": train_sentences,
              additional_input["name"]: np.array(additional_input["data"]).reshape(batch_size,maxlen,1)
            },
            y=train_labels,
            validation_data=(vld_sentences, vld_labels),
            epochs=epochs,
            batch_size=batch_size)
        else:
          self.model.fit(
            train_sentences,
            y=train_labels,
            validation_data=(vld_sentences, vld_labels),
            epochs=epochs,
            batch_size=batch_size)
      else:
        if additional_input:
          self.model.fit(
            {"sentences": train_sentences,
            additional_input["name"]: np.array(additional_input["data"]).reshape(batch_size,maxlen,1)
            },
            y=train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1)
        else:
          self.model.fit(
            {"sentences": train_sentences},
            y=train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1)
      
      # ------ TEST THE MODEL ------
      if test_data:
        if additional_input and additional_input_test:
          self.predict(self.model,
                       test_data, 
                       words_to_index, 
                       maxlen, 
                       self.reverse_label_dict, 
                       additional_input_test)
        else:
          self.predict(self.model, 
                       test_data, 
                       words_to_index, 
                       maxlen, 
                       self.reverse_label_dict)

  def predict(self, model, sentences, words_to_index, maxlen,
              reverse_tagset, additional_input_test=None):
    """Predicts labels for tokens in sentences.
    Args:
      model: the trained model.
      sentences: list of lists. Each list is a sequence of tokens representing a sentence.
    """
    sentence_indices = self.index_sentences(sentences, words_to_index, maxlen)
    if additional_input_test:
      predicate_info =  np.array(
        additional_input_test["data"]
        ).reshape(sentence_indices.shape[0],maxlen,1).astype("float32")

      # the shape of the predictions is (len(sentences), maxlen, n_labels)
      predictions = model.predict(
        {"sentences": sentence_indices,
        "predicate_info": predicate_info
        })
    else:
      predictions = model.predict({"sentences": sentence_indices})

    for i, prediction in enumerate(predictions):
      for j, token_prediction in enumerate(prediction):
        if j < len(sentences[i]):
          print(sentences[i][j], " : ", reverse_tagset[np.argmax(token_prediction)])
      print("----")
  
  def save(self, filename):
    """Serializes the trained model as a json file.
    
    Args:
      filename: str, name to give to the saved model file.
    """
    if not filename.endswith("json"):
      filename += ".json"
    model_dir = "model/nn"
    model_json = self.model.to_json()
    output = os.path.join(model_dir, filename)
    with open(output, "w") as json_file:
      json_file.write(model_json)
    self.model.save_weights(os.path.join(model_dir, 
                                        "{}.h5".format(filename.strip(".json"))))
    logging.info(f"Saved model weights to {model_dir}/{filename.strip('.json')}.h5")
  
  def load(self, filename):
    """Load a pretrained lstm model.
    
    Args:
      filaname: str, name of the model to load.
    """
    model_dir = "model/nn"
    if not filename.endswith(".json"):
      filename += ".json"
    with open(os.path.join(model_dir, filename), "r") as json_file:
      loaded_model = tf.keras.models.model_from_json(json_file.read())
      loaded_model.load_weights(os.path.join(model_dir,
                                             "{}.h5".format(filename.strip(".json"))))
    print(f"Loaded model from {model_dir}/{filename}")
    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    self.model = loaded_model
    print(self.model.summary())
      

# Sample data for a simple pos tagging model.
# ---
# TODO: Remove later.
def get_data():
  maxlen = 6
  train_sentences = [
    ["Can", "bu", "durumda", "sürekli", "kitap", "okur"],
    ["Ali", "getirdiği", "bazı", "kitapları", "bana", "verdi"],
    ["Sana", "verdiğim", "sözleri", "tutacağım"],
    ["Kitabı", "okudum"],
    ["Ali", "eve", "gelmedi"]
  ]
  
  test_sentences = [
    ["burak", "bazı", "durumlarda", "kopya", "çeker"],
    ["bana", "söylediği", "sözleri", "unuttu"],
    ["hep", "geç", "kalır"],
    ["Berna", "devamlı", "resim", "yapar"],
    ["Ali", "getirdiği", "bazı", "kitapları", "bana", "verdi"],
    ["Ona", "verdiğim", "sözleri", "tutacağım"],
    ["Ali", "okula", "gitti"]
  ]
  
  additional_input_train_data = [
    [0, 0, 0, 0, 0 ,1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
  ]
  
  additional_input_test_data = [
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
  ]
  additional_input_train = {"name": "predicate_info",
                            "data": additional_input_train_data,
                            "shape": (maxlen, 1)
                           }
  additional_input_test = {"name": "predicate_info",
                           "data": additional_input_test_data,
                           "shape": (maxlen, 1)
                          }
  return train_sentences, test_sentences, additional_input_train, additional_input_test
    

def get_labels():
  tagset = {"-pad-":0, "nn":1, "prp":2, "nnp":3, "adv":4, "vb":5, "vn":6, "adj":7}
  reverse_tagset = {v:k for k,v in tagset.items()}
  train_tags = [
          ["nnp", "prp", "nn", "adv", "nn", "vb"],
          ["nnp", "vn", "adj", "nn", "prp", "vb"],
          ["prp", "vn", "nn", "vb"],
          ["nn", "vb"],
          ["nnp", "nn", "vb"],
      ]
  return tagset, reverse_tagset, train_tags

# ---


def main(args):
  mylstm = BiLSTM()
  if args.train:
    train_data, test_data, additional_input_train, additional_input_test = get_data()
    label_dict, _, train_labels = get_labels()
    mylstm.train(train_data=train_data,
                 train_data_labels=train_labels, 
                 label_dict=label_dict,
                 epochs=10, 
                 embeddings=True, 
                 loss="categorical_crossentropy", 
                 optimizer="adam",
                 batch_size=5, 
                 test_data=test_data, 
                 test_data_labels=None,
                 additional_input=additional_input_train, 
                 additional_input_test=additional_input_test
                )
    if args.save:
      mylstm.save(args.save_model_name)
  elif args.load:
    mylstm.load(args.load_model_name)
    train_data, test_data, additional_input_train, additional_input_test = get_data()
    label_dict, _, train_data_labels = get_labels()
    word2vec = nn_utils.load_embeddings()
    words_to_index, index_to_words = mylstm.index_words(word2vec)
    maxlen = nn_utils.maxlen(train_data)
    logging.info(f"maxlen: {maxlen}")
    train_sentences = mylstm.index_sentences(train_data, words_to_index, maxlen)
  
    mylstm.label_dict = label_dict
    mylstm.reverse_label_dict = {v:k for k,v in label_dict.items()}
  
    train_labels = mylstm.index_labels(train_data_labels, label_dict, len(train_data), maxlen)
    predicate_info =  np.array(
      additional_input_train["data"]
      ).reshape(train_sentences.shape[0],maxlen,1).astype("float32")
    score = mylstm.model.evaluate({"sentences": train_sentences, "predicate_info": predicate_info}, train_labels, verbose=0)
    print("%s: %.2f%%" % (mylstm.model.metrics_names[1], score[1]*100))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False,
                      help="Trains a new model.")
  parser.add_argument("--save", type=bool, default=False,
                      help="Saves the trained model.")
  parser.add_argument("--save_model_name", type=str,
                      help="Saved model name", default="my_model")
  parser.add_argument("--load", type=bool,
                      help="Loads a model from disk.", default=False)
  parser.add_argument("--load_model_name", type=str,
                      help="Name of the model to load.")
  args = parser.parse_args()
  main(args)
  