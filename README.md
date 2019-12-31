# kBest MST Parser with Argument Structure

__Usage Notes__

- Training a new parser and test on training data
  - Train the model and test on a subportion of the training data. Use --split flag to determine which subpart of the training data will be used for testing.
```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--parser=True \
--language=Turkish \
--train_data=treebank_0_10 \
--split=0.7 \
--split=0.3 \
--epochs=3 \
--parser_model=<model_name_to_save>
```
The above command trains a model using the *treebank_0_10* file under data/UDv23/Turkish directory for 3 epochs. The data is
split into 70% training and 30% testing. The output is saved into a file named "model.json" under mode/pretrained directory.
"model.json" is the default name for saving a file, you can specify a different name using the --model flag. If the flag is not set, the model will be saved to model.json.


- Train a new parser and test on test data
You can train a new parser on a set of data and test it on a separate set of data after the training is finished. To do that, simply specify the path to your test data separately in the --test_data flag.

```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--parser=True \
--language=Turkish \
--train_data=treebank_0_10 \
--test_data=treebank_0_3 \
--epochs=3
```
- Parse sentences with a pretrained model.
You can parse a set of sentences with a pretrained model.

```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=parse \
--language=Turkish \
--load=True \
--parser_model=model.json \
--test_data=treebank_0_3
```
- Evaluate parser/labeler on a set of data.
Use the --evaluate mode to evaluate the parser performance on some parsed data.
You need to have a gold set and a test set, where test set is the parser output
and the gold set is the true annotations.
```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=evaluate \
--gold_data=<path_to_gold_data> \
--test_data=<path_to_test_data> \
--metrics <space_separated_list_of_eval_metrics> \
--print_eval_results=True
```

- Train a labeler and test on test data
You can only train a dependency labeler, or train dependency labeler and dependency parser at the same time.

Note that --learning_rate parameter will only have any meaning for the dependency labeler, not the parser.
The learning rate defaults to 1.0 if not set.
```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--labeler=True \
--language=Turkish \
--train_data=treebank_0_10 \
--test_data=treebank_0_3 \
--epochs=3 \
--learning_rate = 0.5
```

```
bazel build //parser_main:main
bazel-bin/parser_main/main \
--mode=train \
--parser=True \
--labeler=True \
--language=Turkish \
--train_data=treebank_0_10 \
--test_data=treebank_0_3 \
--epochs=3
```

- Train with different feature specs and learning rates
```
bazel-bin/parser_main/main \
--mode=train \
--labeler=True \ # or parser=True or set both to True as separate flags.
--language=Turkish \
--train_data=treebank_tr_imst_ud_train \
--test_data=treebank_tr_imst_ud_test_fixed	\
--epochs=10 \
--learning_rate=1.0 \
--labelfeatures=labelfeatures_exp # and/or arcfeatures=feature_file
```

- Parse and label a treebank at the same time.
bazel-bin/parser_main/main \ 
--load=True \
--parser_model=imst_0_500 \
--labeler_model=test_model_0_50 \
--mode=parse_and_label \
--language=Turkish \
--test_data=treebank_0_3 \
--arcfeatures=arcfeatures_exp7