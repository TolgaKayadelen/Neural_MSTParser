
import tensorflow as tf
import numpy as np

from parser.dep.lfp.label_first_parser import LabelFirstParser
from parser.labeler.bilstm.bilstm_labeler import BiLSTMLabeler
from parser.utils import load_models

def extract_data_for_ranker(labeler_model_name="bilstm_labeler_topk", parser_model_name="label_first_gold_morph_and_labels"):
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  # build labeler
  labeler = BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                          n_output_classes=label_feature.n_values,
                          predict=["labels"],
                          features=["words", "pos", "morph"],
                          model_name=labeler_model_name,
                          top_k=True,
                          k=5,
                          test_every=0)

  # build parser
  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["heads",
                                     # "labels"
                                     ],
                            features=["words",
                                      # "pos",
                                      "morph",
                                      # "category",
                                      # "heads",
                                      "dep_labels"
                                      ],
                            test_every=0,
                            model_name=parser_model_name,
                            one_hot_labels=False)

  # build labeler
  labeler.load_weights(name=labeler_model_name) # uncomment
  for layer in labeler.model.layers:
    print(f"layer name {layer.name}, trainable {layer.trainable}")
  print("labeler ", labeler)

  # build parser
  parser.load_weights(name=parser_model_name)
  for layer in parser.model.layers:
    print(f"layer name {layer.name}, trainable {layer.trainable}")
  print("parser ", parser)
  # build parser

  # get inputs
  test_treebank = "tr_boun-ud-test.pbtxt"
  _, _, test_dataset = load_models.load_data(
    preprocessor=prep,
    test_treebank=test_treebank,
    batch_size=1,
    type="pbtxt",
  )

  total_tokens = 0
  gold_correct = 0
  top1_correct = 0

  # pass inputs through labeler to get top_k outputs
  for batch in test_dataset:
    label_scores, _ = labeler.model({"words": batch["words"], "pos": batch["pos"],
                                      "morph": batch["morph"]})
    top_scores, top_k_labels  = tf.math.top_k(label_scores, k=5)
    top_k_labels = parser._flatten(top_k_labels, outer_dim=top_k_labels.shape[2])
    top_k_labels = tf.cast(top_k_labels, tf.int64)
    # print("top k scores ", top_scores)
    # print("top k lables ", top_k_labels)
    # input()
    correct_labels = batch["dep_labels"]
    correct_labels = parser._flatten(correct_labels)
    # print("correct labels ", correct_labels)
    # input()
    correct_in_topk = tf.reduce_any(correct_labels == top_k_labels, axis=1)
    # print("corr in topk ", correct_in_topk)
    # input()




    scores_with_gold = parser.model({"words":batch["words"], "morph": batch["morph"], "labels": batch["dep_labels"]},
                                    training=False)
    headscores_with_gold = scores_with_gold["edges"]
    heads_with_gold = tf.argmax(headscores_with_gold, axis=2)

    # print("correct labels as input    ", batch["dep_labels"])
    top1_input = tf.expand_dims(top_k_labels[:, 0], 0)
    # print("top 1 labels as input      ", top1_input)
    # print("\n")

    # print("correct heads              ", batch["heads"])
    # print("heads with gold            ", heads_with_gold)

    scores_with_top1 = parser.model({"words":batch["words"], "morph": batch["morph"], "labels": top1_input},
                                    training=False)
    headscores_with_top1 = scores_with_top1["edges"]
    heads_with_top1 = tf.argmax(headscores_with_top1, axis=2)
    # print("heads with top 1           ", heads_with_top1)
    # input()
    # print("\n")
    accuracy_with_gold_labels = np.sum(heads_with_gold == batch["heads"])
    accuracy_with_top1_labels = np.sum(heads_with_top1 == batch["heads"])
    # print("with gold labels           ", accuracy_with_gold_labels)
    # print("with top1 labels           ", accuracy_with_top1_labels)
    # print("\n")
    total_tokens += batch["heads"].shape[1]
    gold_correct += accuracy_with_gold_labels
    top1_correct += accuracy_with_top1_labels
    gold_acc = gold_correct / total_tokens
    top1_acc = top1_correct / total_tokens
    print("total tk ", total_tokens)
    print("gold cor ", gold_correct)
    print("top1 cor ", top1_correct)
    print("gold acc ", gold_acc)
    print("top1 acc ", top1_acc)
    # input()



  # for each k in top_k label outputs
  # pass them through parser to get parser outputs (input is: words, morph, label_indices (for k hypo))
  # you basically call the parser model here (the call function)



  # use the parser outputs to compute the reward for each predicted label.

  # generate features for each label.


  # populate the ranker_data proto


  # when you have reached 100000 exampels save ranker data to file.

if __name__== "__main__":
  extract_data_for_ranker()
