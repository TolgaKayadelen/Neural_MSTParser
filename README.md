# kBest MST Parser with Argument Structure

__Usage Notes__

- Training a new model and test on trianing data
  - Train the model and test on a subportion of the training data. Use --split flag to determine which subpart of the training
  data will be used for testing.
```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--language=Turkish \
--train_data=treebank_0_10 \
--split=0.7 \
--split=0.3 \
--epochs=3
```
The above command trains a model using the *treebank_0_10* file under data/UDv23/Turkish directory for 3 epochs. The data is
split into 70% training and 30% testing. The output is saved into a file named "model.json" under mode/pretrained directory.
"model.json" is the default name for saving a file, you can specify a different name using the --model flag.


- Train a new model and test on test data
You can train a new model on a set of data and test it on a separate set of data after the training is finished. To do
that, simply specify the path to your test data separately in the --test_data flag.

```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--language=Turkish \
--train_data=treebank_0_10 \
--test_data=treebank_0_3 \
--epochs=3
```
- Parse sentences with a pretrained model.
You can parse a set of sentences with at pretrained model.

```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=parse \
--language=Turkish \
--load=model.json \
--test_data=treebank_0_3
```
- Evaluate parser on a set of data.
Use the --evaluate mode to evaluate the parser performance on some parsed data.
You need to have a gold set and a test set, where test set is the parser output
and the gold set is the true annotations.
```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=evaluate \
--gold_data=<path_to_gold_data> \
--test_data=<path_to_test_data> \
--metrics <space_separated_list_of_eval_metrics>
```
