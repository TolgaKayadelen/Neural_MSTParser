"""Extracts datapoint protos for the reranker from treebanks."""

import os
import logging

import tensorflow as tf
import numpy as np

from math import log
from parser.dep.lfp.label_first_parser import LabelFirstParser
from parser.labeler.bilstm.bilstm_labeler import BiLSTMLabeler
from parser.utils import load_models
from ranker.preprocessing import feature_extractor
from proto import ranker_data_pb2
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import reader, writer

_feature_extractor = feature_extractor.FeatureExtractor()

def _enumerated_tensor(_tensor):
  """Converts a 2D tensor to its enumerated version."""
  enumerated_tensors_list = []
  if not len(_tensor.shape) == 2:
    raise ValueError(f"enumerated tensor only works for 2D tensors. Received tensor of shape {_tensor.shape}")
  batch_size = _tensor.shape[0]
  for i in range(batch_size):
    # print("i ", i)
    # print("tensor[i]", _tensor[i])
    # input("press ")
    _t = tf.constant([i, tf.keras.backend.get_value(_tensor[i][0])])
    enumerated_tensors_list.append(_t)

  _enumerated_tensor = tf.convert_to_tensor(enumerated_tensors_list)
  # print("enumerated_tensor ", _enumerated_tensor)
  # input("press")
  return _enumerated_tensor

def beam_search_decoder(data, k=2):
  """Returns k best sequences over a 2D matrix based on beam search.

  Args:
    data: a 2D np.array
    k: size of beam
  """
  sequences = [[list(), 0.0]]
  # walk over each step in sequence
  for row in data:
    # print(f"row is {row}")
    # input()
    all_candidates = list()
    # expand each current candidate
    for i in range(len(sequences)):
      # print(f"sequences {sequences}")
      # input()
      seq, score = sequences[i]
      # print(f"seq is {seq}")
      for j in range(len(row)):
        # print(f"j is {j}")
        candidate = [seq + [j], score - log(row[j])]
        # print(f"candidate is {candidate}")
        all_candidates.append(candidate)
    # order all candidates by score
    # print("all candidates ", all_candidates)
    # input()
    ordered = sorted(all_candidates, key=lambda tup:tup[1])
    # print(f"ordered {ordered}")
    # input()
    # select k best
    sequences = ordered[:k]
  return sequences



def reward_for_hypothesis_prioritize_edges(label, gold_label, head_score):
  """Computes reward for a label hypothesis. This reward function prioritizes edges over labels.

    The reward ranges between -1 and 2, such that:

      If head_correct and label_correct: reward = 3
      If head_correct and label_false: reward = 2
      If head_false and label_correct: reward = 1
      If head_false and label_false: reward = 0

  Returns:
    reward: float.
  """
  # print(f"hypothesis label: {label}, gold label: {gold_label}, head score: {head_score}")
  label_correct = False
  reward = 10.0
  if label == gold_label:
    label_correct = True
  if head_score == 1:
    reward += 10.0
    if label == gold_label:
      reward += 10.0             # when both edge and label is correct, reward is 3.
    else:
      reward -= 0.0             # when edge is correct but label is not, reward equals 2.
  else:
    if label == gold_label:
      reward += 0.0             # when edge is false but label correct, reward is 2.
    else:
      reward -= 10.0             # if both edge and label is false, reward 0.
  # print("reward is ", reward)
  return reward


def reward_for_hypothesis_only_edges(label, gold_label, head_score):
  """Computes reward for a label hypothesis. This reward function only considers edges..

      If head_correct: reward = 10
      Else: reward = -10

  Returns:
    reward: float.
  """
  # print(f"hypothesis label: {label}, gold label: {gold_label}, head score: {head_score}")

  reward = 0.0
  if head_score == 1:
    reward += 1.0
  return reward

def reward_for_hypothesis(label, gold_label, head_score):
  """Computes reward for a label hypothesis.

    The reward ranges between -1 and 2, such that:

      If head_correct and label_correct: reward = 3
      If head_correct and label_false: reward = 1
      If head_false and label_correct: reward = 2
      If head_false and label_false: reward = 0

  Returns:
    reward: float.
  """
  # print(f"hypothesis label: {label}, gold label: {gold_label}, head score: {head_score}")
  label_correct = False
  reward = 100
  if label == gold_label:
    label_correct = True
  if head_score == 1:
    reward += 100.0
    if label == gold_label:
      reward += 100.0             # when both edge and label is correct, reward is 3.
    else:
      reward -= 100.0             # when edge is correct but label is not, reward equals 1.
  else:
    if label == gold_label:
      reward += 100.0             # when edge is false but label correct, reward is 2.
    else:
      reward -= 100.0             # if both edge and label is false, reward 0.
  # print("reward is ", reward)
  return reward

def ranker_datapoint(word_ids, gold_labels, tokens, top_k_labels, head_scores):
  """Creates a list of ranker datapoints from a sentence.
  Args:
    word_ids: the words (embedding ids) in the sentence.
    gold_labels: the gold dependency label for each word.
    tokens = list of token_pb2.token objects in this sentence
    top_k_labels: the top k label predictions for each token.
    head_scores: the attachment score with the parser for each of the top k labels.

  Returns:
    a list of ranker_pb2.RankerDatapoint() objects.
  """
  datapoints = []
  for i, word in enumerate(word_ids[0]):
    dp = ranker_data_pb2.RankerDatapoint()

    word_id = word_ids[:, i][0]
    # print("word_id", word_id)

    # word = words[:, i][0]
    token = tokens[i]
    # print("word ", word)
    word = token.word
    # print("word ", word)
    # print("token proto ", token)
    # input()


    top_k_for_token = top_k_labels[i, :]
    # print("top k for this token ", top_k_for_token)


    gold_label = gold_labels[:, i][0]
    # print("gold label ", gold_label)
    # print("head scores for this token with each top k labels", head_scores[i])
    # input()

    dp.word_id = tf.keras.backend.get_value(word_id)
    dp.word = tf.keras.backend.get_value(word)
    dp.features.CopyFrom(_feature_extractor.get_features(token, tokens, n_prev=-2, n_next=2))
    for k in range(top_k_labels.shape[1]):
      hypothesis_k = tf.keras.backend.get_value(top_k_for_token[k])
      hypothesis = dp.hypotheses.add()
      hypothesis.label = dep_label_tags.Tag.Name(hypothesis_k)
      hypothesis.label_id = hypothesis_k
      hypothesis.rank = k+1
      hypothesis.reward = reward_for_hypothesis_only_edges(hypothesis.label_id, gold_label, head_scores[i][k])
    # print("ranker datapoint ", dp)
    # input()
    datapoints.append(dp)
  return datapoints

def generate_dataset_for_ranker(*, labeler, parser, dataset, treebank_path, k=5, beam_search=False):
  """Generates a training dataset for the reranker.
  Args:
    labeler: A pretrained labeler.
    parser: A pretrained parser.
    dataset: The treebank to generate the training dataset from. A tf.data.Datase object.
  """
  ranker_dataset = ranker_data_pb2.RankerDataset()
  treebank = reader.ReadTreebankTextProto(treebank_path)
  sentences = {sentence.sent_id:sentence.token for sentence in treebank.sentence}
  # print("sentences ", sentences)

  total_tokens = 0
  gold_correct = 0
  top1_correct = 0
  beam_correct = 0
  counter = 0
  for example in dataset:
    counter += 1
    if counter % 100 == 0:
      print(f"processing sentence {counter}")
    # pass inputs through labeler to get top_k outputs
    scores = labeler.model({"words": example["words"], "pos": example["pos"], "morph": example["morph"]})
    label_scores = scores["labels"]
    top_scores, top_k_labels  = tf.math.top_k(label_scores, k=k)
    top_k_labels = parser._flatten(top_k_labels, outer_dim=top_k_labels.shape[2])
    top_k_labels = tf.cast(top_k_labels, tf.int64)
    # print("top k scores ", top_scores)
    # print("Top k labels ", top_k_labels)
    correct_labels = example["dep_labels"]
    correct_labels = parser._flatten(correct_labels)
    # print("correct labels ", correct_labels)
    correct_in_topk = tf.reduce_any(correct_labels == top_k_labels, axis=1)
    # print("corr in topk ", correct_in_topk)
    total_tokens += example["heads"].shape[1]
    # input()

    sent_id = tf.keras.backend.get_value(example['sent_id'][0][0])
    tokens = sentences[sent_id.decode("utf-8")]
    # logging.info(f"Sentence ID: {sent_id}, Tokens: {example['tokens']}")
    # logging.info(f"Token protos {tokens}")
    # input()

    # Get the head accuracy scores using the gold labels as input. This is just for bookkeeping later.
    scores_with_gold = parser.model({"words": example["words"], "morph": example["morph"],
                                     "labels": example["dep_labels"]}, training=False)
    headscores_with_gold = scores_with_gold["edges"]
    heads_with_gold = tf.argmax(headscores_with_gold, axis=2)
    accuracy_with_gold_labels = np.sum(heads_with_gold == example["heads"])
    gold_correct += accuracy_with_gold_labels
    gold_acc = gold_correct / total_tokens

    # for each k in top_k label outputs, pass them through parser to get parser outputs
    k_best_head_scores = []
    for i in range(top_k_labels.shape[1]):
      top_i = tf.expand_dims(top_k_labels[:, i], 0)
      # print("top {} labels: {:>}".format(i+1, str(top_i)))
      # print("gold labels : {:>11}".format(str(example["dep_labels"])))
      scores_with_i = parser.model({"words": example["words"], "morph": example["morph"],
                                    "labels": top_i}, training=False)
      headscores_with_i = scores_with_i["edges"]
      heads_with_i = tf.argmax(headscores_with_i, axis=2)
      # print("heads with top {}           : {:>10}".format(i+1, str(heads_with_i)))
      # print("correct heads              : {:>10}".format(str(example["heads"])))
      # print("heads with gold            : {:>10}".format(str(heads_with_gold)))
      # print("\n\n")
      correct_preds_with_i = (heads_with_i == example["heads"])
      # print("correct preds with top {}   : {:>10}".format(i+1, str(correct_preds_with_i)))

      accuracy_with_top_i = np.sum(correct_preds_with_i)
      # print("accuracy with top {}        : {:>10}".format(i+1, accuracy_with_top_i))
      # input()

      # Keep records of accuracy with top label hypotheses. Again this is for later comparison only.
      if i == 0:
        top1_correct += accuracy_with_top_i
        top1_acc = top1_correct / total_tokens

      k_best_head_scores.append(correct_preds_with_i)
      
    # Convert k_best head scores to float and transpose them to a shape that can be used to generate data.
    k_best_head_scores = np.squeeze(np.array(k_best_head_scores).astype(float)).transpose()
    # print("k best head scores ", k_best_head_scores)
    # input()

    ranker_dps = ranker_datapoint(example["words"], example["dep_labels"], tokens, top_k_labels,
                                  tf.convert_to_tensor(k_best_head_scores))
    for dp in ranker_dps:
      datapoint = ranker_dataset.datapoint.add()
      datapoint.CopyFrom(dp)

    if beam_search:
      beam_correct += compute_performance_with_beam_search(k_best_head_scores, top_k_labels, parser)
      beam_acc = beam_correct / total_tokens
      print("totak tk ", total_tokens)
      print("gold corr ", gold_correct)
      print("beam corr ", beam_correct)
      print("top 1 corr ", top1_correct)
      print("gold acc ", gold_acc)
      print("beam acc ", beam_acc)
      print("top1 acc ", top1_acc)

    # print("Ranker Dataset ", ranker_dataset)
    # input()
  
  return ranker_dataset

def compute_performance_with_beam_search(k_best_head_scores, top_k_labels, parser):
  k_best_head_scores[k_best_head_scores == 0] = 0.001
  print("k best array ", k_best_head_scores)
  print("shape ", k_best_head_scores.shape)

  # Run beam search over the top_k labels to determine the best sequences.
  best_sequences = beam_search_decoder(k_best_head_scores, 5)
  print("best sequences ", best_sequences)
  scores = []

  # Select labels from top_k labels based on results of the beam search.
  # then parse with them to see what's the head accuracy.
  # TODO: implement logic to determine both head and label accuracy with the beam labels,
  # TODO: currently it only computes head accuracy and doesn't factor in head accuracy.
  # TODO: and select the k-best beam label sequence based on las > uas > ls.
  for i, sequence in enumerate(best_sequences):
    # print("seq is ", sequence)
    # print(sequence[0])
    seq_index = tf.expand_dims(tf.convert_to_tensor(sequence[0]), 1)
    # print("seq index ", seq_index)
    enumerated = _enumerated_tensor(seq_index)
    print("enumerated ", enumerated)
    labels_collected = tf.expand_dims(tf.gather_nd(top_k_labels, enumerated), 0)
    print("labels collected ", labels_collected)
    scores_col = parser.model({"words":batch["words"], "morph": batch["morph"], "labels": labels_collected},
                               training=False)
    headscores_with_col = scores_col["edges"]
    heads_with_col = tf.argmax(headscores_with_col, axis=2)
    correct_preds_col = (heads_with_col == batch["heads"])
    print("correct head preds after beam search ", correct_preds_col)
    scores.append((i, labels_collected, np.sum(correct_preds_col)))
  print("scores after beam ", scores)
  beam_correct = scores[0][2]
  return beam_correct

if __name__== "__main__":
  _data_dir = "./data/UDv29/dev/tr"
  _ranker_data_dir = "./ranker/data"
  _treebank_name = "tr_boun-ud-dev.pbtxt"

  labeler_model_name="bilstm_labeler_topk"
  parser_model_name="label_first_gold_morph_and_labels"

  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  labeler=load_models.load_labeler(labeler_name=labeler_model_name, prep=prep)
  # labeler=load_models.load_parser(labeler_model_name, predict=["heads", "labels"],
  #                                 features=["words", "pos", "morph"], prep=prep)
  parser=load_models.load_parser(parser_name=parser_model_name, prep=prep)

  # get inputs
  train_dataset, dev_dataset, test_dataset = load_models.load_data(
    preprocessor=prep,
    dev_treebank=_treebank_name,
    dev_batch_size=1,
    type="pbtxt",
  )
  # for batch in dev_dataset:
  #   print(batch)
    # input()
  treebank_path = os.path.join(_data_dir, _treebank_name)
  ranker_dataset = generate_dataset_for_ranker(labeler=labeler, parser=parser,
                                               treebank_path=treebank_path,
                                               dataset=dev_dataset,
                                               k=20)

  writer.write_proto_as_text(ranker_dataset, "./ranker/data/tr_boun-ud-dev-k20-only-edges-dp.pbtxt")
  logging.info("Ranker dataset written to //ranker/data/tr_boun-ud-dev-k20-only-edges-dp.pbtxt ")
