#!/bin/sh

test_data=(treebank_0_10 treebank_0_10)
epochs=(1 2)
for i in "${test_data[@]}"
do
  for j in "${epochs[@]}"
  do
    bazel-bin/parser_main/main --mode=train --parser=True --language=Turkish --train_data=$i \
    --split=0.9 --split=0.1  --epochs=$j --learning_rate=1.0 --arcfeatures=arcfeatures_exp2;
  done
done