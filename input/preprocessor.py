import tensorflow as tf
import numpy as np
import random
from enum import Enum
from tagset.reader import LabelReader
from typing import List, Dict, Generator, Union
from util.nn import nn_utils
from util import reader
import time

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Array = np.ndarray
Dataset = tf.data.Dataset
SequenceExample = tf.train.SequenceExample


words = np.array([
       [1,152339,380947,484432,340375,536702,1,0,0,0,0,0,0,0,0],
       [1,34756, 224906, 578174, 506596,1,0,0,0,0,0,0,0,0,0],
       [1,119951,1,562326,490947,305574,359585,453123,444258,1,0,0,0,0,0],
       [1,570228,548366,507341,361412,220841,474903,319297,334201,258062,380947,396760,559275,508964,1]]
       )

pos = np.array([
       [30, 14, 28, 31,  4,  3, 24,  0,  0,  0,  0,  0,  0,  0,  0],
       [30,  5, 14, 14, 31, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [30, 23, 24, 14, 14, 31, 20,  4, 31, 24,  0,  0,  0,  0,  0],
       [30, 14, 31,  3, 14, 14, 14, 14, 14, 14, 28, 14, 14, 31, 24]])

targets = np.array(
[[[0, 1],
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1],
  [0, 1],
  [1, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0]],
 [[0, 1],
  [0, 1],
  [0, 1],
  [0, 1],
  [1, 0],
  [0, 1],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0]],
 [[0, 1],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0]],
 [[1, 0],
  [1, 0],
  [0, 1],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [0, 1]]])
  
# To test training with variable shapes.
words = np.array([
       [1,152339,380947,484432,340375,536702,1],
       [1,34756, 224906, 578174, 506596,1],
       [1,119951,1,562326,490947,305574,359585,453123,444258,1],
       [1,570228,548366,507341,361412,220841,474903,319297,334201,258062,380947,396760,559275,508964,1]]
       )

pos = np.array([
       [30, 14, 28, 31,  4,  3, 24],
       [30,  5, 14, 14, 31, 24],
       [30, 23, 24, 14, 14, 31, 20,  4, 31, 24],
       [30, 14, 31,  3, 14, 14, 14, 14, 14, 14, 28, 14, 14, 31, 24]])

targets = np.array(
[[[0, 1],
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1],
  [0, 1],
  [1, 0]],
 [[0, 1],
  [0, 1],
  [0, 1],
  [0, 1],
  [1, 0],
  [0, 1]],
 [[0, 1],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1]],
 [[1, 0],
  [1, 0],
  [0, 1],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1],
  [0, 1]]])

class SanityCheck(Enum):
  UNKNOWN = 1
  PASS = 2
  FAIL = 3

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self, name: str, values: List[int]=[], dtype=tf.int64,
               one_hot: bool = False, n_values: int = None,
               is_label: bool = False):
    self.name = name
    self.values = values
    self.dtype = dtype
    self.n_values = n_values
    self.one_hot = one_hot
    self.is_label = is_label
  
  def _convert_to_one_hot(self, values, n_values):
    """Converts an integer array one hot vector.
    
    An integer array of shape (1, m) becomes a one hot vector of shape
    (len(indices), n_labels).
    """
    if isinstance(values, list):
      logging.info("Converting list to array")
      values = np.array(values)
    if not isinstance(values, np.ndarray):
      raise ValueError(f"""Can only convert numpy arrays to one hot vectors,
                      received: {type(values)}""")
    return np.eye(n_values)[values.reshape(-1)]
  
  @property
  def one_hot(self):
    return self._one_hot
  
  @one_hot.setter
  def one_hot(self, one_hot):
    if one_hot:
      if not self.n_values:
        raise RuntimeError("n_values must be defined for one hot conversion.")
      logging.info("Converting values to one hot objects""")
      self._one_hot = self._convert_to_one_hot(self.values, self.n_values)
    else:
      self._one_hot = one_hot
    return self._one_hot
  
  def __str__(self):
    return f"""name: {self.name},
            values: {self.values},
            one_hot : {self.one_hot},
            n_values: {self.n_values}"""


class Embeddings:
  """Class to initialize embeddings from a pretrained embedding matrix."""
  def __init__(self, *, name:str, matrix):
    self.name = name
    self.vocab = matrix
    self.vocab_size = len(self.vocab)
    self.embedding_dim = matrix
    self.token_to_index = self._token_to_index()
    self.index_to_token = self._index_to_token()
    self.token_to_vector = matrix
    self.index_to_vector = self._index_to_vector(matrix=matrix)
  
  def __str__(self):
    return f"""
      {self.name} embeddings, dimension: {self.dimension},
      vocab_size: {self.vocab_size}
      """

  def _token_to_index(self) -> Dict[int, int]:
    return {elem:ind for ind, elem in enumerate(self.vocab)}
  
  def _index_to_token(self) -> Dict[int, int]:
    return {ind:elem for ind, elem in enumerate(self.vocab)}       

  def _index_to_vector(self, *, matrix) -> Array:
    index_to_vector = np.zeros(shape=(self.vocab_size, self.embedding_dim))
    for token, idx in self.token_to_index.items():
      # print(token, idx)
      if token == "-pad-":
        index_to_vector[idx, :] = np.zeros(shape=(self.embedding_dim,))
      elif token == "-oov-":
        # randomly initialize
        index_to_vector[idx, :] = np.random.randn(self.embedding_dim,)
      else:
        index_to_vector[idx, :] = matrix[token]
        # print(matrix[token])
    return index_to_vector
  
  # Accessor methods for individual items are defined here.
  def stoi(self, *, token: str) -> int:
    return self.token_to_index[token]
  
  def itos(self, *, idx: int) -> str:
    return self.index_to_token[idx]
  
  def ttov(self, *, token: str):
    return self.token_to_vector[token]
  
  def itov(self, *, idx: int):
    return self.index_to_vector[idx]
  
  # Embedding properties are set based on the properties of the 
  # word2vec matrix.
  @property
  def vocab(self):
    return self._vocab
  
  @vocab.setter
  def vocab(self, matrix) -> List[str]:
    "Setting up the vocabulary."
    print("Setting up the vocabulary")
    vocab = []
    vocab.extend(["-pad-", "-oov-"])
    vocab.extend(sorted(set(matrix.wv.vocab.keys())))
    self._vocab = vocab
  
  @property
  def embedding_dim(self):
    return self._embedding_dim
  
  @embedding_dim.setter
  def embedding_dim(self, matrix):
    print("Setting embedding dimension")
    random_token = random.choice(self.vocab)
    self._embedding_dim = matrix[random_token].shape[0]
  
  @property
  def sanity_check(self) -> SanityCheck:
    print("Running sanity checks on the created embeddings")
    random_token = random.choice(self.vocab)
    random_tok_idx = self.stoi(token=random_token)
    if not len(self.token_to_vector.wv.vocab.keys()) == self.vocab_size - 2:
      return SanityCheck.FAIL
    if not self.vocab[0] == "-pad-" and self.vocab[1] == "-oov-":
      return SanityCheck.FAIL
    if not self.stoi(token=random_token) == self.token_to_index[random_token]:
      return SanityCheck.FAIL
    if not self.itos(idx=random_tok_idx) == self.index_to_token[random_tok_idx]:
      return SanityCheck.FAIL
    if not (self.token_to_vector[random_token] == self.index_to_vector[
      random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.itov(idx=random_tok_idx) == self.index_to_vector[
      random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.token_to_vector[self.itos(idx=random_tok_idx)
      ] == self.token_to_vector[random_token]).all():
      return SanityCheck.FAIL
    if not (self.index_to_vector[self.stoi(token=random_token)
      ] == self.token_to_vector[random_token]).all():
      return SanityCheck.FAIL
    return SanityCheck.PASS


class Preprocessor:
  """Prepares data as batched tfrecords."""
  def __init__(self, *, word_embeddings: Embeddings = None):
    if word_embeddings:
      self.word_embeddings = word_embeddings
  
  def numericalize(self, *, values: List[str],
                   mapping: Union[Dict, Embeddings]) -> List[int]:
    """Returns numeric values (integers) for features based on a mapping."""
    if isinstance(mapping, Embeddings):
      indices = []
      for value in values:
        try:
          indices.append(mapping.stoi(token=value))
        except KeyError:
          indices.append(mapping.stoi(token="-oov-"))
      return indices
    elif isinstance(mapping, dict):
      return [mapping[value] for value in values]
    else:
      raise ValueError("mapping should be a dict or an Embedding instance.")
  
  def _get_index_mappings_for_features(self, feature_names: List[str]) -> Dict:
    "Returns a string:index map for a feature based on predefined configs"
    mappings = {}
    if "words" in feature_names:
      if self.word_embeddings:
        mappings["words"] = self.word_embeddings
      else:
        embeddings = nn_utils.load_embeddings()
        mappings["words"] = Embeddings(name="word2vec", matrix=embeddings)
    if "morph" in feature_names:
      raise RuntimeError("Morphology features are not supported yet.")
    if "pos" in feature_names:
      mappings["pos"] = LabelReader.get_labels("pos").labels
    if "category" in feature_names:
      mappings["category"] = LabelReader.get_labels("category").labels
    if "srl" in feature_names:
      raise RuntimeError("Semantic role features are not supported yet.")
    return mappings
        
  # TODO: this needs to be updated to read one hot labels if the 
  # feature is a label feature.
  def make_tf_example(self, *, features: List[SequenceFeature]
                     ) -> SequenceExample:
    """Creates a tf.train.SequenceExample from a single datapoint.
    
    A single datapoint is represented as a List of SequenceFeature objects.
    """
    example = tf.train.SequenceExample()
    
    for feature in features:
      feature_list = example.feature_lists.feature_list[feature.name]
      for token in feature.values:
        feature_list.feature.add().int64_list.value.append(token)
    
    return example
    
  def write_tf_records(self, *, examples: List[SequenceExample], path: str
                      ) -> None:
    """Serializes tf.train.SequenceExamples to tfrecord files."""
    filename = path + ".tfrecords" if not path.endswith(".tfrecords") else path
    writer = tf.io.TFRecordWriter(filename)
    for example in examples:
      writer.write(example.SerializeToString())
    writer.close()
  
  # TODO: this needs to be udpated to read the feature shapes properly when
  # the feature is a label   
  def make_dataset_from_tfrecords(self, *,
                                  batch_size: int = 2,
                                  features: List[SequenceFeature],
                                  records: str) -> Dataset:
    """Makes a tensorflow dataset from a saved tfrecords path."""
    
    _sequence_features = {}
    _dataset_shapes = {}
    
    # Create a dictionary description of the tensors to parse the features.
    for feature in features:
      _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature([],
                         dtype=feature.dtype)
      _dataset_shapes[feature.name]=tf.TensorShape([None])
    
    def _parse_tf_records(record):
      """Returns a dictionary of tensors."""
      _, parsed_example = tf.io.parse_single_sequence_example(
        serialized=record,
        sequence_features=_sequence_features
      )
      return {feature.name:parsed_example[feature.name] for feature in features}
    
    dataset = tf.data.TFRecordDataset([records])
    dataset = dataset.map(_parse_tf_records)    
    dataset = dataset.padded_batch(batch_size, padded_shapes=_dataset_shapes)
    return dataset
    
  def make_dataset_from_generator(self, *, path: str, batch_size: int = 2,
                                  features=List[SequenceFeature],
                                  generator: Generator=None) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded.
    Args:
      path: path to the dataset treebank. 
      generator: a generator function.
                 If not specified, the class generator is used.
    """
    if not generator:
      generator = lambda: self._example_generator(path, features)
    
    _output_types={}
    _output_shapes={}
    _padded_shapes={}
    
    for feature in features:
      _output_types[feature.name]=feature.dtype
      
      if feature.is_label:
        _output_shapes[feature.name]=(None, feature.n_values)
        _padded_shapes[feature.name]=tf.TensorShape((None, feature.n_values))
      else:
        _output_shapes[feature.name]=[None]
        _padded_shapes[feature.name]=tf.TensorShape([None])

    
    dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=_output_types,
      output_shapes=_output_shapes
    )

    dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.padded_batch(batch_size, padded_shapes=_padded_shapes)
    # dataset = dataset.padded_batch(2, padded_shapes=(_padded_shapes, tf.TensorShape((None,2))))
    print(dataset)
    return dataset
  

  def _example_generator(self, path: str, features:List[SequenceFeature]):
    trb = reader.ReadTreebankTextProto(path)
    sentences = trb.sentence
    feature_names = [feature.name for feature in features]
    feature_mappings = self._get_index_mappings_for_features(feature_names)
    label_feature = next((f for f in features if f.is_label), None)
    yield_dict = {}
    
    for sentence in sentences:
      for feature_name in feature_names:
        if feature_name == "words":
          yield_dict[feature_name] = self.numericalize(
            values=[token.word for token in sentence.token],
            mapping=feature_mappings["words"])
        if feature_name == "pos":
          yield_dict[feature_name] = self.numericalize(
            values=[token.pos for token in sentence.token],
            mapping=feature_mappings["pos"])
        if feature_name == "category":
          yield_dict[feature_name] = self.numericalize(
            values=[token.category for token in sentence.token],
            mapping=feature_mappings["category"])
      # print(yield_dict["pos"])
      for k in yield_dict:
        if k == label_feature.name:
          labels = np.array([label_feature.one_hot[np.where(
            label_feature.one_hot[:, item] == 1)][0] for item in yield_dict[k]])
          yield_dict[k] = labels
          break
      # print(yield_dict["pos"])
      yield yield_dict



class BiLSTM:
  
  def __init__(self, *, word_embeddings: Embeddings, n_classes):
    self.word_embeddings_layer = self._set_pretrained_embeddings(word_embeddings)
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.model = self._model(n_classes)
    self.optimizer = tf.keras.optimizers.Adam(0.001)
    self.train_metrics = tf.keras.metrics.CategoricalAccuracy()
    self.val_metrics = tf.keras.metrics.CategoricalAccuracy()

  def _set_pretrained_embeddings(self, embeddings: Embeddings):
    """Builds a pretrained keras embedding layer from an Embeddings object."""
    embed = tf.keras.layers.Embedding(embeddings.vocab_size,
                                      embeddings.embedding_dim,
                                      weights=[embeddings.index_to_vector],
                                      trainable=False,
                                      name="word_embeddings")
    embed.build((None,))
    return embed
  
  def __str__(self):
    return str(self.model.summary())
  
  def _model(self, n_classes):
    word_inputs = tf.keras.Input(shape=([None]), name="words")
    cat_inputs = tf.keras.Input(shape=([None]), name="cat")
    cat_features = tf.keras.layers.Embedding(32, 32,
                                             name="cat_embeddings")(cat_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    concat = tf.keras.layers.concatenate([cat_features, word_features])
    encoded1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                              units=150,
                                              return_sequences=True,
                                              name="encoded"))(concat)
    encoded1 = tf.keras.layers.Dropout(rate=0.5)(encoded1)                                          
    encoded2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                              units=75,
                                              return_sequences=True,
                                              name="encoded"))(encoded1)                                          
    X = tf.keras.layers.Dense(units=n_classes)(encoded2)
    # NOTE: as you have from_logits=True in your loss function. You don't use the activation layer.
    # X = tf.keras.layers.Activation("softmax", name="result")(X)
    
    model = tf.keras.Model(inputs=[word_inputs, cat_inputs], outputs=X)
    tf.keras.utils.plot_model(model, "multi_input_and_output_model.png",
                              show_shapes=True)
    
    # NOTE: You don't compile if you use custom low level training.
    # model.compile(
    #  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
    #  loss="categorical_crossentropy",
    #  metrics=["acc"]
    #)
    return model
  
  @tf.function
  def train_step(self, words, cat, y):
    # print("y shape ", y.shape)
    # input("press to cont")
    with tf.GradientTape() as tape:
      logits = self.model([words, cat], training=True)
      # print("logits shape ", logits.shape)
      # input("press to cont")
      # Send the **LOGITS** to the loss function.
      loss_value = self.loss_fn(y, logits)
    grads = tape.gradient(loss_value, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    # Update train metrics
    self.train_metrics.update_state(y, logits)
    return loss_value
  
  @tf.function
  def test_step(self, x, y):
    val_logits = self.model(x["words"], x["pos"], training=False)
    self.val_metrics.update_state(y, val_logits)
  
  def train(self, dataset, epochs=40):
    for epoch in range(epochs):
      # Reset training metrics at the end of epoch
      self.train_metrics.reset_states()
      print("==========================================")
      logging.info(f"Training epoch {epoch}")
      start_time = time.time()
      for step, batch in enumerate(dataset):
        words = batch["words"]
        cat = batch["category"]
        y = batch["pos"]
        loss_value = self.train_step(words, cat, y)
        # print(loss_value)
        
        # Log
        # if step % 10 == 0: 
        #  logging.info(f"Training loss for one batch at step {step} :  {float(loss_value)}")
      
      # Display metrics at the end of each epoch
      train_acc = self.train_metrics.result()
      logging.info(f"Training accuracy over epoch {train_acc}")
      
      
      # Log the time it takes for one epoch
      logging.info(f"Time for epoch: {time.time() - start_time}")
  

      
    
if __name__ == "__main__":
  
  # Load word embeddings
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  
  # Initialize the preprocessor.
  prep = Preprocessor(word_embeddings=word_embeddings)
  
  """
  # MAKING DATASET BY SAVING AND READING TFRECORDS.
  examples = []
  # Making dataset from tfrecords.
  pos_mapping = LabelReader.get_labels("pos").labels
  print(f"pos mapping: {pos_mapping}")
  cat_mapping = LabelReader.get_labels("category").labels
  print(f"cat mapping: {cat_mapping}")
  datapath = "data/UDv23/Turkish/training/treebank_0_3.pbtxt"
  trb = reader.ReadTreebankTextProto(datapath)
  # Make tf examples
  for sentence in trb.sentence:
    word_indices = prep.numericalize(values=[token.word for token in sentence.token],
                                     mapping=prep.word_embeddings
    )
    words = SequenceFeature(name="words", values=word_indices)
    pos_indices = prep.numericalize(values=[token.pos for token in sentence.token],
                                    mapping=pos_mapping)
    postags = SequenceFeature(name="pos", values=pos_indices)
    example = prep.make_tf_example(features=[words, postags])
    examples.append(example)
  
  # Write examples as tf records.
  prep.write_tf_records(examples=examples, path="./input/test1.tfrecords")
  
  # Make dataset from saved tfrecords
  features = [SequenceFeature(name="words"), SequenceFeature(name="pos")]
  dataset = prep.make_dataset_from_tfrecords(features=features,
                                             records="./input/test1.tfrecords")
  """
  # MAKING DATASET FROM GENERATOR.
  
  pos_dict = LabelReader.get_labels("pos").labels
  pos_tag_indices = list(pos_dict.values()) 
  pos_feature = SequenceFeature(name="pos", values=pos_tag_indices,
                                n_values=len(pos_tag_indices),
                                is_label=True, one_hot=True)

  dataset = prep.make_dataset_from_generator(
    path="data/UDv23/Turkish/training/treebank_0_3.pbtxt",
    features=[SequenceFeature(name="words"), SequenceFeature(name="category"),
    pos_feature])
  # for batch in dataset:
  #  print(batch)
  mylstm = BiLSTM(word_embeddings=prep.word_embeddings,
                  n_classes=pos_feature.n_values)
  print(mylstm)
  mylstm.train(dataset)
  
  # You can use dataset directly as np.arrays rather than dataset API.
  # In this case you give the dataset as a dict to the custom training function.
  # And you modify the training funtion to make sure that it accepts values as dict. 
  # dataset={"words": words, "pos": pos, "targets": targets}
  # mylstm.train(dataset)
  # mylstm.model.fit([words, pos], y=targets, epochs=30)

  