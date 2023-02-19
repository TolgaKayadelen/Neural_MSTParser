import io
import os
import json
import gensim
import matplotlib.pyplot as plt
import numpy as np
import random
from util import reader, writer
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from proto import metrics_pb2
from collections import defaultdict, OrderedDict, Counter

_EMBEDDING_DIR = "embeddings"

def plot_metrics(*, name: str, metrics: metrics_pb2.Metrics, plot_losses=False):
  """Plots the metrics from a parser run."""
  fig = plt.figure()
  ax = plt.axes()
  loss_metrics = ["edge_loss_padded", "label_loss_padded"] 
  colors = ["b", "g", "r", "c", "m", "y", "sienna", "orchid", "k"]
  for key in metrics.metric:
    if key in loss_metrics and not plot_losses:
      continue
    if metrics.metric[key].tracked:
      color = random.choice(colors)
      colors.remove(color)
      ax.plot(np.arange(len(metrics.metric[key].value_list.value)),
              metrics.metric[key].value_list.value,
              "-g", label=key, color=color)
  plt.title("Neural MST Parser Performance")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(f"./model/nn/plot/{name}_metrics_plot")

def set_up_metrics(*args) -> metrics_pb2.Metrics:
  metrics = metrics_pb2.Metrics()
  for metric in args:
     metrics.metric[metric].tracked = True
  print("metrics ", metrics)
  # input()
  return metrics

def convert_to_one_hot(indices, n_labels):
  """Converts an integer array of shape (1, m) to a one hot vector of shape (len(indices), n_labels)
  
  Args:
    indices: An integer array representing the labels or classes.
    n_labels: depth of the one hot vector.
  Returns:
    one hot vector of shape (len(indices), n_labels)
  """
  return np.eye(n_labels)[indices.reshape(-1)]
  

def load_embeddings(name="tr-word2vec-model_v3.bin"):
  """Loads a pretrained word embedding model into memory."""
  word2vec_bin = os.path.join(_EMBEDDING_DIR, name)
  word2vec = gensim.models.Word2Vec.load(word2vec_bin)
  
  return word2vec


def maxlen(data):
  """Returns the length of the longest list in a list of lists."""
  return len(max(data, key=len))


def annotate_bio_spans(sentence):
  """Annotatest BIO spans in a sentence.
  
  This function gets a sentence where semantic role labels are only
  marked on the immediate child tokens of the verb and propagates these
  labels using the BIO tagging framework to all the children of those
  tokens. 
  
  Example:
    input = The boy_AO loves the dog_A1 very much.
    output = The_B-AO boy_I-AO loves the_B-A1 cat_B-A1 very_O much_O.
  
  Args:
    sentence: a sentence_pb2.Sentence object.
  Returns:
    sentence: a sentence_pb2.Sentence object.
  
  """
  heads = defaultdict(list)
  srls = Counter()
  for token in sentence.token:
    if token.index == 0: 
      continue
    heads[token.selected_head.address].append(token)
  heads = OrderedDict(sorted(heads.items(), reverse=True))
  
  for head in heads.keys():
    if not sentence.token[head].HasField("srl"):
      continue
    children = heads[head]
    for child in children:
      child.srl = sentence.token[head].srl

  for token in sentence.token:
    if token.index == 0:
      continue
    if not token.HasField("srl"):
      token.srl = "O"
    else:
      if srls[token.srl] == 0:
        srls[token.srl] += 1
        token.srl = "B-" + token.srl
      else:
        token.srl = "I-" + token.srl    

  return sentence
  

def get_argument_spans(sentence, token_index, predicate_index, argument_span=[]):
  """Returns all the argument span of an argument based on the head.
  
  Args:
    sentence_pb2.Sentence()
    token_index = the index of the token in the sentence.
  Returns:
    span_indices: set of token indices which represent the argument span.
  """
  # Special treatment for "case" tokens because of the weird way
  # they are annotated in Turkish propbank.
  if sentence.token[token_index].label == "case":
    head_address = sentence.token[token_index].selected_head.address
    head_of_span = sentence.token[head_address].index
    if not head_of_span == predicate_index:
      if not sentence.token[head_of_span] in argument_span:
        argument_span.append(sentence.token[head_of_span])
  else:
    head_of_span = token_index

  for token in sentence.token:
    if token in argument_span:
      continue
    if token.index == predicate_index:
      continue
    if token.category == "PUNCT":
      continue
    if token.selected_head.address == head_of_span:
      argument_span.append(token)
      get_argument_spans(sentence, token.index, predicate_index, argument_span)
  span_indices = set([token.index for token in argument_span])
  return list(span_indices)


def morph_label_cooccurences(treebank):
  cooccurences = defaultdict(Counter)
  sentences = treebank.sentence
  total_tokens = sum([sentence.length for sentence in sentences])
  total_subjects = 0
  total_objects = 0
  for sentence in sentences:
    for token in sentence.token:
      morphology = token.morphology
      for morph in morphology:
        if morph.name == "case":
          cooccurences[morph.value][token.label] += 1
  return cooccurences


def create_class_weight(labels_dict, mu=0.9):
  total = np.sum(list(labels_dict.values()))
  keys = labels_dict.keys()
  class_weight = dict()

  for key in keys:
    score = math.log(mu*total/float(labels_dict[key]))
    class_weight[key] = score if score > 1.0 else 1.0
  return class_weight

def generate_multilingual_embeddings(language):
  dir = f"./embeddings/{language}"
  file = os.path.join(dir, f"cc.{language}.300.vec")
  print("file ", file)
  fin = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
  # n, d = map(int, fin.readline().split())
  data = {}
  counter = 0
  for line in fin:
    counter += 1
    tokens = line.rstrip().split(' ')
    if counter == 1:
      print(tokens[0])
      continue
    data[tokens[0]] = tokens[1:]
    if counter % 10000 == 0:
      print(counter)
    if counter == 500000:
      break
  json_file = os.path.join(dir, f"{language}_embeddings.json")
  with open(json_file, 'w') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    json.dump(data, f)
  return data

def load_pickled_embeddings(language):
  with open(f'./embeddings/{language}/{language}_embeddings.json', 'rb') as f:
    return json.load(f)

def compute_embedding_occurence(language="fi",
                                filepath="./data/UDv29/languages/Finnish/UD_Finnish-TDT/fi_tdt-ud-train.txt"):
  matrix = load_pickled_embeddings(language=language)
  keys = matrix.keys()
  total_keys = len(keys)
  print("total keys ", total_keys)
  fi = open(filepath, "r")
  lines = fi.readlines()
  n_found, n_found_unique, n_unique, n_words = 0, 0, 0, 0
  found_words, unique_words = [], []
  for line in lines:
    words = line.split()
    for word in words:
      n_words += 1
      word = word.strip(".,'")
      if not word in unique_words:
        unique_words.append(word)
        n_unique += 1
      if word in keys:
        print("word found ", word)
        n_found += 1
        if not word in found_words:
          found_words.append(word)
          n_found_unique += 1
  print("total words ", n_words)
  print("total found ", n_found)
  print("total unique ", n_unique)
  print("total found unique ", n_found_unique)
  print("ratio found words ", n_found / n_words)
  print("ratio unique words that are found ", n_found_unique / n_unique)

# for splitting propbank sentenes to predicate-sequence datapoints.
def generate_multisentence_data(treebank_path):
  treebank = reader.ReadTreebankTextProto(treebank_path)
  new_treebank = treebank_pb2.Treebank()
  new_sentences = []
  sentences = treebank.sentence
  for sentence in sentences:
    counter = 0
    for arg_str in sentence.argument_structure:
      counter += 1
      new_sentence = sentence_pb2.Sentence()
      new_sentence.CopyFrom(sentence)
      new_sentence.sent_id = f"{sentence.sent_id}-{counter}"
      for token in new_sentence.token:
        token.ClearField("srl")
        # token.srl = "-0-"
        token.predicative = 0
      predicate_token = new_sentence.token[arg_str.predicate_index]
      predicate_token.predicative = 1
      for argument in arg_str.argument:
        # print("argument ", argument)
        # print(argument.token_index)
        # print(type(argument.token_index))
        # input()
        argument_token = new_sentence.token[argument.token_index[0]]
        srl = argument_token.srl.append(argument.srl)
        # argument_token.srl.add(argument.srl)
      for token in new_sentence.token:
        if len(token.srl) > 0:
          continue
        token.srl.append("-0-")
      new_sentences.append(new_sentence)
  for sentence in new_sentences:
    s = new_treebank.sentence.add()
    s.CopyFrom(sentence)
  writer.write_proto_as_text(new_treebank, f"{treebank_path}_multisent")


def generate_merged_data(treebank_path):
  treebank = reader.ReadTreebankTextProto(treebank_path)
  new_treebank = treebank_pb2.Treebank()
  new_sentences = []
  sentences = treebank.sentence
  print("total sentence ", len(sentences))
  for sentence in sentences:
    new_sentence = sentence_pb2.Sentence()
    new_sentence.CopyFrom(sentence)
    for token in new_sentence.token:
      token.ClearField("srl")
      # token.srl = "-0-"
      token.predicative = 0
    for arg_str in sentence.argument_structure:
      predicate_token = new_sentence.token[arg_str.predicate_index]
      predicate_token.predicative = 1
      for argument in arg_str.argument:
        # print("argument ", argument)
        # print(argument.token_index)
        # print(type(argument.token_index))
        # input()
        argument_token = new_sentence.token[argument.token_index[0]]
        srl = argument_token.srl.append(argument.srl)
        # argument_token.srl.add(argument.srl)
    for token in new_sentence.token:
      if len(token.srl) > 0:
        continue
      token.srl.append("-0-")
    new_sentences.append(new_sentence)
    # print(new_sentence)
  for sentence in new_sentences:
    s = new_treebank.sentence.add()
    s.CopyFrom(sentence)
  writer.write_proto_as_text(new_treebank, f"{treebank_path}_merged")




if __name__ == "__main__":
  # generate_multilingual_embeddings("en")
  treebank_path = "./data/propbank/ud/srl/dev.pbtxt"
  generate_merged_data(treebank_path)
  #trb = reader.ReadTreebankTextProto(
  #  "data/testdata/propbank/propbank_ud_testdata_proto.pbtxt"
  #)
  #sentence = trb.sentence[1]
  #annotate_bio_spans(sentence)

  # print(words)
  # print(np.array(matrix["und"], dtype=np.float32))

