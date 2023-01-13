import os
import csv
from util import reader
from tagset.reader import LabelReader

_DATA_DIR = "./data/UDv29/train/tr"
_DEV_DATA_DIR = "./data/UDv29/dev/tr"
_TEST_DATA_DIR = "./data/UDv29/test/tr"
_TRAINING_FILE = "tr_boun-ud-train.pbtxt"
_DEV_FILE = "tr_boun-ud-dev.pbtxt"
_TEST_FILE ="tr_boun-ud-test.pbtxt"


_dep_labels = LabelReader("dep_labels", "tr", False).labels

def _write_csv(file, treebank):
  with open(f'./transformer/hf/data/{file}.csv', mode='w') as csv_file:
    fieldnames = ['sent_id', 'tokens', 'dep_labels']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


    writer.writeheader()
    for sentence in treebank.sentence:
      sent_id = sentence.sent_id
      tokens = str([token.word for token in sentence.token])
      tokens=tokens.replace("[","")
      tokens=tokens.replace("]","")
      dep_labels = str([_dep_labels[token.label] for token in sentence.token])
      dep_labels=dep_labels.replace("[","")
      dep_labels=dep_labels.replace("]","")
      writer.writerow({'sent_id': sent_id, 'tokens': tokens, 'dep_labels': dep_labels})

def create_datasets():
  datafiles = {
    "train": os.path.join(_DATA_DIR, _TRAINING_FILE),
    "dev": os.path.join(_DEV_DATA_DIR, _DEV_FILE),
    "test": os.path.join(_TEST_DATA_DIR, _TEST_FILE),
  }
  for datafile in datafiles.keys():
    treebank = reader.ReadTreebankTextProto(datafiles[datafile])
    _write_csv(datafile, treebank)
  # treebank = reader.ReadTreebankTextProto(datafiles["test"])
  # _write_csv("test", treebank)

if __name__ == "__main__":
  create_datasets()