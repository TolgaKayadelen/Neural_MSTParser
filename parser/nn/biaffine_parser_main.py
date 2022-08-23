
import os
import datetime
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn.biaffine_parser import BiaffineParser
from parser.nn import load_models

if __name__ == "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/biaffine_boun/" + current_time
  # tf.debugging.experimental.enable_dump_debug_info(
  #  log_dir,
  #  tensor_debug_mode="FULL_HEALTH",
  #  circular_buffer_size=-1)
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiaffineParser(word_embeddings=prep.word_embeddings,
                          n_output_classes=label_feature.n_values,
                          predict=["heads", "labels"],
                          features=["words", "pos", "morph"],
                          model_name="biaffine_parser_boun",
                          test_every=5)

  train_treebank= "tr_boun-ud-train.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  train_dataset, dev_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                                   train_treebank=train_treebank,
                                                                   batch_size=250,
                                                                   test_treebank=test_treebank,
                                                                   type="pbtxt")
  metrics = parser.train(dataset=train_dataset, epochs=100,
                         test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/final/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")
