
import sys
import logging
import datetime

import numpy as np

from input import preprocessor
from parser.utils import load_models
from parser.dep.lfp.bert_label_first_parser import BertLabelFirstParser
from tagset import reader
from util import writer


if __name__ == "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  label_embedding_weights = load_models.load_layer_weights("label_embeddings_weights")
  print(f"Loaded label embedding weights from pretrained with shape: {label_embedding_weights.shape}")
  pos_embeddings_weights = load_models.load_layer_weights("pos_embeddings_weights")
  print(f"Loaded pos embedding weights from pretrained with shape {pos_embeddings_weights.shape}")
  # input()
  word_embeddings = load_models.load_word_embeddings()

  parser_model_name = "bert_label_first_parser"

  logging.info(f"Parser model name is {parser_model_name}")

  # model_name_check = input("Did you remember to set the model name properly: y/n?")

  # if model_name_check != "y":
  #   raise ValueError("Model name is not set properly! Please set your model name and restart!")

  log_dir = f"debug/label_first_parser/{parser_model_name}/{current_time}"

  prep = preprocessor.Preprocessor(
    features=["pos", "morph", "heads", "dep_labels", "sent_id"],
    labels=["heads"],
  )

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  num_labels = label_feature.n_values
  print(f"num labels {num_labels}")

  parser = BertLabelFirstParser(name=parser_model_name,
                                word_embeddings=word_embeddings,
                                log_dir=log_dir,
                                num_labels=num_labels,
                                test_every=5,
                                label_embedding_weights=label_embedding_weights,
                                pos_embeddings_weights=pos_embeddings_weights,
                                # pretrained_bert_model_path="./transformer/hf/pretrained/bert-lfp-parser/20230122-060042/bert",
                                # load_pretrained="./transformer/hf/pretrained/bert-lfp-parser/20230122-060042"
                                )



  # get the data
  # train_treebank= "tr_boun-ud-train-random10.pbtxt"

  # test_treebank = "tr_boun-ud-test-random2.pbtxt"
  # train_dataset, _, test_dataset = load_models.load_data(preprocessor=prep,
  #                                                        train_treebank=train_treebank,
  #                                                        batch_size=2,
  #                                                        test_treebank=test_treebank,
  #                                                        type="pbtxt",
  #                                                        language="tr")
  # for batch in train_dataset:
  #   print(batch)
  # input()
  parser.train(epochs=1)
  # print(metrics)
  # writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
  # parser.save_weights()
  # logging.info("weights saved!")
