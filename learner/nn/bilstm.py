"""A BiLSTM sequence model."""

import tensorflow as tf
import numpy as np
import gensim

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class BiLSTM:
  """A BiLSTM model for sequence labeling."""
  def __init__(self, embeddings=True):
    self.pretrain_embeddings = embeddings
  
  def pretrained_embeddings_layer(self):
    """Creates an embedding layer for the Neural Net and feeds a pretrained word2vec model into it.
    
    Returns:
      embedding_layer. A pretrained Keras Embedding() layer.
    """
    word2vec_bin = '/Users/tolgakayadelen/Desktop/Thesis/tr-parsing/neural_mst_dependency_parser/embeddings/tr-word2vec-model_v3.bin' 
    word2vec = gensim.models.Word2Vec.load(word2vec_bin)
    words_to_index, index_to_words = self.words_to_index(word2vec)
    # print(words_to_index["tolga"]) # should return 493047
    
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
    
  def words_to_index(self, embedding_model):
    """Turns all the vocabulary items in an embedding model into indices.
    
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
    
    
if __name__ == "__main__":
  mylstm = BiLSTM()
  embedding_layer = mylstm.pretrained_embeddings_layer()
  print(embedding_layer.get_weights()[0][490735])

  