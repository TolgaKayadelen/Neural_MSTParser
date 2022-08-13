import datetime
from parser.nn import load_models
from parser.nn.label_first_parser_exp import LabelFirstParser
from util import writer

if __name__ == "__main__":
  # use_pretrained_weights_from_labeler = True
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/label_first_parser/" + current_time
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)

  parser_model_name = "lfp_with_coarse_dep_labels"
  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=8,
                            predict=["heads",
                                     # "labels"
                                     ],
                            features=["words",
                                      "pos",
                                      "morph",
                                      "category",
                                      "heads",
                                      "dep_labels"
                                      ],
                            log_dir=log_dir,
                            test_every=5,
                            model_name=parser_model_name,
                            one_hot_labels=False)


  # get the data
  train_treebank= "tr_boun-ud-train-random500.pbtxt"
  test_treebank = "tr_boun-ud-test-random50.pbtxt"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=5,
                                                      test_treebank=test_treebank,
                                                      test_batch_size=1,
                                                      type="pbtxt")


  # for batch in train_dataset:
  #   print(batch)
  # input()
  metrics = parser.train(dataset=train_dataset, epochs=75, test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
  # parser.save_weights()
  # logging.info("weights saved!")
