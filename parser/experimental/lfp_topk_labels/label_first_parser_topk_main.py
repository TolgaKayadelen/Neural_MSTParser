import logging
import numpy as np
import datetime

from parser.utils import load_models
from parser.experimental.lfp_topk_labels import label_first_parser_topk
from util import writer
from util.nn import nn_utils

if __name__ == "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  pretrained_parser_model_name = "label_first_gold_morph_and_labels"
  labeler_model_name="bilstm_labeler_topk"
  parser_model_name="label_first_parser_topk"
  log_dir = f"debug/label_first_parser_topk/{parser_model_name}/{current_time}"

  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  pretrained_parser = load_models.load_parser(pretrained_parser_model_name, prep)
  label_embeddings = False
  for layer in pretrained_parser.model.layers:
    if layer.name == "label_embeddings":
      label_embedding_weights = layer.get_weights()
      label_embeddings=True
      break
  logging.info("Retrieved label embeddings")
  if not label_embeddings:
    raise NameError("Parser doesn't have a layer named label_embeddings.")

  # get the data
  train_treebank= "tr_boun-ud-train.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  train_dataset, _, test_dataset = load_models.load_data(preprocessor=prep,
                                                         train_treebank=train_treebank,
                                                         batch_size=250,
                                                         test_treebank=test_treebank,
                                                         type="pbtxt",
                                                         language="tr")

  parser = label_first_parser_topk.LabelFirstParserTopk(word_embeddings=word_embeddings,
                                                        preprocessor=prep,
                                                        predict = ["heads"],
                                                        features =["words", "pos", "morph", "dep_labels"],
                                                        log_dir=log_dir,
                                                        test_every=5,
                                                        model_name=parser_model_name,
                                                        labeler_name=labeler_model_name,
                                                        label_embedding_weights=label_embedding_weights)

  metrics = parser.train(dataset=train_dataset, epochs=75, test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
  parser.save_weights()
  logging.info("weights saved!")
