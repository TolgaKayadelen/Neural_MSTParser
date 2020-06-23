"""A BiLSTM sequence model."""

import os
import tensorflow as tf
import numpy as np
import gensim


from util import writer
from learner.nn.utils import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class BiLSTM:
  """A BiLSTM model for sequence labeling."""
  def __init__(self, embeddings=True):
    self.pretrain_embeddings = embeddings
    
  def pretrained_embeddings_layer(self, word2vec, words_to_index):
    """Creates an embedding layer for the Neural Net and feeds a pretrained word2vec model into it.
    
    Args:
      word_to_index: dictionary where keys are words and values are indices.
      word2vec: a pretrained word embeddings model.
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
    embedding_dim = word2vec[index_to_words[100]].shape[0]
    
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
      sentences: array of sentences, shape(m, 1) where m is the number of sentences.
      word_indices: dictionary containing each word mapped to an index.
      maxlen: scalar, maximum number of words in a sentence.
    
    Returns:
      sentence_indices: array of indices corresponding of each word in each sentence
        of shape (m, maxlen)
    
    Sentences which have shorter than maxlen words are padded with 0s on the right.
    """
    
    m = sentences.shape[0]
    
    sentence_indices = np.zeros(shape=(m, maxlen))
    for i in range(m):
      words_in_sentence = sentences[i].split()
      
      j = 0
      
      for w in words_in_sentence:
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
        padded = np.pad(indexed, (0, pad_value), "constant", constant_values=(tagset["-pad-"]))
        return padded
      return indexed
  
    label_indices = np.zeros(shape=(n_sentences, maxlen, len(label_dict)))
    for i, label_set in enumerate(labels):
      indexed = _labels_to_index(label_set)
      print(i)
      print(indexed)
      one_hot = nn_utils.convert_to_one_hot(indexed, len(tagset))
      print(one_hot)
      label_indices[i, :, :] = one_hot
    return label_indices
    

  def model(self, input_shape, word2vec, word_indices, n_classes):
    """Creates a BiLSTM model.
    
    Args:
      input_shape: shape of the input, usually (max_len, )
      word2vec: the word embeddings model.
      word_indices: dict, key:value pairs of words and indices.
    
    Returns:
      a Keras Model instance.
    """
    
    # Define sentence_indices as the Input to the graph. It has dtype=int32
    # because it contains indices for each word in the sentence, which are
    # integers.
    sentence_indices = tf.keras.layers.Input(shape=input_shape, dtype="int32")
    
    # Create the embedding layer with the word2vec model.
    embedding_layer = self.pretrained_embeddings_layer(word2vec, word_indices)
    
    # Propagate sentences through the embedding layer.
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128 hideen units.
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(embeddings)
    
    # Add dropout.
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    
    # Propagate through another LSTM layer.
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(X)
    
    # Add dropout.
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    
    # Add the Dense Layer.
    X = tf.keras.layers.Dense(units=n_classes)(X)
    # Add activation
    X = tf.keras.layers.Activation("softmax")(X)
    
    # Create the model.
    model = tf.keras.Model(inputs=sentence_indices, outputs=X)
    
    return model


  def predict(self, model, sentences, words_to_index, maxlen, reverse_tagset):
    for sentence in sentences:
      print(sentence)
      sentence_list = sentence.split()
      sentence_array = np.array([sentence])
      sentence_indices = self.index_sentences(sentence_array, words_to_index, maxlen)
      predictions = model.predict(sentence_indices)
      for i, prediction in enumerate(predictions[0]):
        if i < len(sentence_list):
          print(sentence_list[i], " : ", reverse_tagset[np.argmax(prediction)])



# Sample data for a simple pos tagging model.
# ---
# TODO: Remove later.
def get_data():
  sentences = [
    "Can bu durumda sürekli kitap okur",
    "Ali getirdiği bazı kitapları bana verdi",
    "Sana verdiğim sözleri tutacağım",
    "Kitabı okudum",
    "Ali eve gelmedi"
  ]
  return sentences
    

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

if __name__ == "__main__":
  mylstm = BiLSTM()
  word2vec = nn_utils.load_embeddings()
  words_to_index, index_to_words = mylstm.index_words(word2vec)
  train_data = get_data()
  maxlen = nn_utils.maxlen(train_data)
  sentences = np.array(train_data)
  sentence_indices = mylstm.index_sentences(sentences, words_to_index, maxlen)
  # print(sentences, sentence_indices)
  
  tagset, reverse_tagset, train_labels = get_labels()
  postagger = mylstm.model(input_shape=(maxlen,),
                           word2vec=word2vec,
                           word_indices=words_to_index,
                           n_classes=len(tagset))
  print(postagger.summary())
  postagger.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  labels = mylstm.index_labels(train_labels, tagset, sentences.shape[0], maxlen)

  print(f"shape of the output {labels.shape}")
  
  postagger.fit(sentence_indices, labels, epochs=30, batch_size=5)
  
  test_sentences = [
    "burak bazı durumlarda kopya çeker",
    "bana söylediği sözleri unuttu",
    "hep geç kalır",
    "Berna devamlı resim yapar"
  ]
  mylstm.predict(postagger, test_sentences, words_to_index, maxlen, reverse_tagset)

  