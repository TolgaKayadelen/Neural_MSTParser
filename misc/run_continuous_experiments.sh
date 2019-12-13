#!/bin/sh

# test_data=(treebank_train_0_500 treebank_train_500_1000 treebank_train_1000_1500 treebank_train_1500_2000 treebank_train_2000_2500)
epochs=(10 20 30)
#for i in "${test_data[@]}"
#do
  for j in "${epochs[@]}"
  do
    bazel-bin/parser_main/main --mode=train --parser=True --language=Turkish --train_data=treebank_tr_imst_ud_train \
    --test_data=treebank_tr_imst_ud_test_fixed  --epochs=$j --learning_rate=1.0 --arcfeatures=arcfeatures_exp2;
  done
#done
