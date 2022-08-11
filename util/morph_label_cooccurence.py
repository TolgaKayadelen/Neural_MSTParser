import os
from util import reader
from collections import defaultdict, Counter

_DATA_DIR="./data/UDv29/train/tr"

def morph_label_cooccurences(treebank):
  cooccurences = defaultdict(Counter)
  sentences = treebank.sentence
  total_tokens = sum([sentence.length for sentence in sentences])
  total_subjects = 0
  total_objects = 0
  for sentence in sentences:
    for token in sentence.token:
      morphology = token.morphology
      for morph in morphology:
        if morph.name == "case":
            cooccurences[morph.value][token.label] += 1

  return cooccurences

if __name__ == "__main__":
  treebank_name = "tr_boun-ud-train.pbtxt"
  treebank = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, treebank_name))
  cooccurences = morph_label_cooccurences(treebank)
  for case in cooccurences.keys():
    print(case)
    for k in cooccurences[case]:
      print(k, cooccurences[case][k])
    print("-----")
