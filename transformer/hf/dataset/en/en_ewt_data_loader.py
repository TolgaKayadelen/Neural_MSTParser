"""Script to load dep_label dataset for HF transformer library."""

import re
import os
import pickle
import datasets
import logging
from copy import deepcopy

_DESCRIPTION = """English EWT treebank"""
_DATA_DIR = "./data/UDv29/languages/English/UD_English-EWT"
_TRAINING_FILE = "en_ewt-ud-train.conllu"
_DEV_FILE = "en_ewt-ud-dev.conllu"
_TEST_FILE ="en_ewt-ud-test.conllu"


def read_conllx(path):
  """Read treebank from a file where sentences are in conll-X format.
  Args:
      corpus: path to the treebank file.
  Returns:
      list of sentences each in Conll-x format.
  """

  fi = open(path, "r")
  lines = fi.readlines()
  sentence_lines = []
  sentence_list = []
  sentence_counter = 0

  # the main loop
  for line in lines:
    if line != "\n":
      sentence_lines.append(line)
    else:
      sentence_counter += 1
      sentence = deepcopy(sentence_lines)
      sentence_list.append(sentence)
      del sentence_lines[:]

  logging.debug("Read %d sentences!" % sentence_counter)
  fi.close()
  return sentence_list

# This method can be used to test with the main function whether the generator is working properly.
def convert_to_dict(sentence_list):
  """Converts a conll-X formatted sentence to a dictionary.

  Args:
      sentence: a sentence in conll-X format.
      semantic_roles: if True, also adds semantic role label to the token.
  Returns:
      sentence mapped to a defaultdict.
      metadata: the text and the id of the sentence.
  """
  for sentence in sentence_list:
    tokens = []
    heads = []
    dep_labels = []
    for line in sentence:
      if line.startswith('# newdoc'):
        continue
      if line.startswith('# Change'):
        continue
      if line.startswith("# Fixed"):
        continue
      if line.startswith("# Fix"):
        continue
      if line.startswith("# NOTE"):
        continue
      if line.startswith('# Checktree'):
        continue
      if line.startswith('# global'):
        continue
      if line.startswith('# s_type'):
        continue
      if line.startswith("# meta"):
        continue
      if line.startswith("# newpar"):
        continue
      if line.startswith("# speaker"):
        continue
      if line.startswith("# addressee"):
        continue
      if line.startswith("# trailing_xml"):
        continue
      if line.startswith("# TODO"):
        continue
      if line.startswith("# orig"):
        continue
      if line.startswith("# sent_id"):
        sent_id = line.split("=")[1].strip()
        continue
      if line.startswith("# text"):
        continue
      if re.match("(([0-9]|[1-9][0-9]|[1-9][0-9][0-9])-([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))", line):
        continue
      if re.match("([0-9][0-9]\.[0-9]|[0-9]\.[0-9])", line):
        continue
      values = [item.strip() for item in line.split("\t")]
      tokens.append(values[1])
      heads.append(values[6])
      dep_labels.append(values[7])
    yield {
      "sent_id": sent_id,
      "tokens": tokens,
      "heads": heads,
      "dep_labels": dep_labels,
    }

class EnglishEWTTreebankConfig(datasets.BuilderConfig):
  """BuilderConfig for BounDepLabels"""

  def __init__(self, **kwargs):
    """BuilderConfig for BounTreebank.
    Args:
      **kwargs: keyword arguments forwarded to super.
    """
    super(EnglishEWTTreebankConfig, self).__init__(**kwargs)


class EnglishEWTTreebank(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
      EnglishEWTTreebankConfig(name="EnglishEWTTreebank", version=datasets.Version("1.0.0"),
                               description="English EWT Dependency Treebank"),
    ]
    def _info(self):
      return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
          {
            "sent_id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "heads": datasets.Sequence(datasets.Value("int32")),
            "dep_labels": datasets.Sequence(
              datasets.features.ClassLabel(
                names=[
                  'TOP',
                  "acl",
                  "acl:relcl",
                  "advcl",
                  "advmod",
                  "amod",
                  "appos",
                  "aux",
                  "aux:pass",
                  "case",
                  "cc",
                  "cc:preconj",
                  "ccomp",
                  "compound",
                  "compound:prt",
                  "conj",
                  "cop",
                  "csubj",
                  "csubj:pass",
                  "dep",
                  "det",
                  "det:predet",
                  "discourse",
                  "dislocated",
                  "expl",
                  "fixed",
                  "flat",
                  "flat:foreign",
                  "goeswith",
                  "iobj",
                  "list",
                  "mark",
                  "nmod",
                  "nmod:npmod",
                  "nmod:poss",
                  "nmod:tmod",
                  "nsubj",
                  "nsubj:pass",
                  "nummod",
                  "obj",
                  "obl",
                  "obl:npmod",
                  "obl:tmod",
                  "orphan",
                  "parataxis",
                  "punct",
                  "reparandum",
                  "root",
                  "vocative",
                  "xcomp",
                ]
              )
            ),
          }
        ),
        supervised_keys=None
      )

    def _split_generators(self, dl_manager):
      """Returns split generators"""
      datafiles = {
        "train": os.path.join(_DATA_DIR, _TRAINING_FILE),
        "dev": os.path.join(_DATA_DIR, _DEV_FILE),
        "test": os.path.join(_DATA_DIR, _TEST_FILE),
      }

      return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": datafiles["train"]}),
        datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": datafiles["dev"]}),
        datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": datafiles["test"]}),
      ]

    def _generate_examples(self, filepath):
      logging.info(f"Generating examples from {filepath}")
      sentence_list = read_conllx(filepath)
      key = 0
      for sentence in sentence_list:
        tokens = []
        heads = []
        dep_labels = []
        key += 1
        for line in sentence:
          if line.startswith('# newdoc'):
            continue
          if line.startswith('# Change'):
            continue
          if line.startswith("# Fixed"):
            continue
          if line.startswith("# Fix"):
            continue
          if line.startswith("# NOTE"):
            continue
          if line.startswith('# Checktree'):
            continue
          if line.startswith('# global'):
            continue
          if line.startswith('# s_type'):
            continue
          if line.startswith("# meta"):
            continue
          if line.startswith("# newpar"):
            continue
          if line.startswith("# speaker"):
            continue
          if line.startswith("# addressee"):
            continue
          if line.startswith("# trailing_xml"):
            continue
          if line.startswith("# TODO"):
            continue
          if line.startswith("# orig"):
            continue
          if line.startswith("# text"):
            continue
          if line.startswith("# sent_id"):
            sent_id = line.split("=")[1].strip()
            continue
          if re.match("(([0-9]|[1-9][0-9]|[1-9][0-9][0-9])-([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))", line):
            continue
          if re.match("([0-9][0-9]\.[0-9]|[0-9]\.[0-9])", line):
            continue
          values = [item.strip() for item in line.split("\t")]
          tokens.append(values[1])
          heads.append(values[6])
          dep_labels.append(values[7])
        yield key, {
          "sent_id": sent_id,
          "tokens": tokens,
          "heads": heads,
          "dep_labels": dep_labels,
        }

if __name__ == "__main__":
  sentence_list = read_conllx(os.path.join(_DATA_DIR, _DEV_FILE))
  converted = convert_to_dict(sentence_list)
  for sentence in converted:
    print(sentence)
    input()
