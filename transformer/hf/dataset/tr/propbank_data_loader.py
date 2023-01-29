"""Script to load dep_label for propbank"""

import re
import os
import pickle
import datasets
import logging
from copy import deepcopy

_DESCRIPTION = """IMST propbank"""
_DATA_DIR = "./data/propbank/ud"
_TRAINING_FILE = "merged.conllu.train"
_DEV_FILE = "merged.conllu.dev"
_TEST_FILE ="merged.conllu.test"

replace_dict = {
  "Postp": "PostP",
  "PcAbl": "PCAbl",
  "Adv": "Adverb",
}

def load_embedding_indexes():
  with open('./transformer/hf/dataset/tr/token_to_index_dictionary.pkl', 'rb') as f:
    return pickle.load(f)

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
# You can implement this part in the data laoder if you want to generate SRL dataset for BERT.
def convert_to_dict(sentence_list):
  """Converts a conll-X formatted sentence to a dictionary.

  Args:
      sentence: a sentence in conll-X format.
      semantic_roles: if True, also adds semantic role label to the token.
  Returns:
      sentence mapped to a defaultdict.
      metadata: the text and the id of the sentence.
  """
  token_to_index = load_embedding_indexes()
  counter = 0
  sent_id_template = "sent_id"
  for sentence in sentence_list:

    input()
    counter += 1
    tokens = []
    token_ids = []
    pos = []
    heads = []
    dep_labels = []
    predicates = []
    srls = []
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
      else:
        sent_id = f"{sent_id_template}_{counter}"
      if line.startswith("# text"):
        continue
      if re.match("(([0-9]|[1-9][0-9]|[1-9][0-9][0-9])-([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))", line):
        continue
      if re.match("([0-9][0-9]\.[0-9]|[0-9]\.[0-9])", line):
        continue
      values = [item.strip() for item in line.split("\t")]
      total_values = len(values)
      print("total values ", total_values)
      input()
      tokens.append(values[1])
      try:
        token_ids.append(token_to_index[values[1]])
      except KeyError:
        print("token ", values[1])
        token_ids.append(1)
      if values[4] in ["Postp", "PcAbl", "Adv"]:
        pos.append(replace_dict[values[4]])
      else:
        pos.append(values[4])
      heads.append(values[6])
      dep_labels.append(values[7])
      srl_range = [i+12 for i in range(total_values-12)]
      if values[10] == "Y":
        predicates.append("PREDICATE:YES")
      else:
        predicates.append("PREDICATE:NO")
      print("srl range ", srl_range)
      srl = "-0-"
      for i in srl_range:
        if values[i] != "_":
          srl = values[i]
          break
      srls.append(srl)

    yield {
      "sent_id": sent_id,
      "tokens": tokens,
      "token_ids": token_ids,
      "pos": pos,
      "heads": heads,
      "dep_labels": dep_labels,
      "predicates": predicates,
      "srls": srls
    }

class PropbankConfig(datasets.BuilderConfig):
  """BuilderConfig for BounDepLabels"""

  def __init__(self, **kwargs):
    """BuilderConfig for BounTreebank.
    Args:
      **kwargs: keyword arguments forwarded to super.
    """
    super(PropbankConfig, self).__init__(**kwargs)


class BounTreebank(datasets.GeneratorBasedBuilder):
  BUILDER_CONFIGS = [
    PropbankConfig(name="TrPropbank", version=datasets.Version("1.0.0"),
                   description="Tr IMST Propbank"),
  ]
  def _info(self):
    return datasets.DatasetInfo(
      description=_DESCRIPTION,
      features=datasets.Features(
        {
          "sent_id": datasets.Value("string"),
          "tokens": datasets.Sequence(datasets.Value("string")),
          "token_ids": datasets.Sequence(datasets.Value("int32")),
          "pos": datasets.Sequence(datasets.Value("string")),
          "heads": datasets.Sequence(datasets.Value("int32")),
          "dep_labels": datasets.Sequence(
            datasets.features.ClassLabel(
              names=[
                'TOP',
                'acl',
                'advcl',
                'advmod',
                'advmod:emph',
                'amod',
                'appos',
                'aux',
                'aux:q',
                'case',
                'cc',
                'ccomp',
                'compound',
                'compound:lvc',
                'compound:redup',
                'conj',
                'cop',
                'csubj',
                'dep',
                'det',
                'discourse',
                'fixed',
                'flat',
                'mark',
                'nmod',
                'nmod:poss',
                'nsubj',
                'nummod',
                'obj',
                'obl',
                'parataxis',
                'punct',
                'root',
                'vocative',
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
    if filepath.endswith(".train"):
      sent_id_template = "train_sentence"
    elif filepath.endswith(".dev"):
      sent_id_template = "dev_sentence"
    elif filepath.endswith(".test"):
      sent_id_template =  "test_sentence"
    else:
      raise ValueError("Couldn't figure out dataset type!")
    # print("sent id template ", sent_id_template)
    # input()
    token_to_index = load_embedding_indexes()
    sentence_list = read_conllx(filepath)
    key = 0
    counter = 0
    for sentence in sentence_list:
      counter += 1
      tokens = []
      token_ids = []
      pos = []
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
        else:
          sent_id = f"{sent_id_template}_{counter}"
        if re.match("(([0-9]|[1-9][0-9]|[1-9][0-9][0-9])-([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))", line):
          continue
        if re.match("([0-9][0-9]\.[0-9]|[0-9]\.[0-9])", line):
          continue
        values = [item.strip() for item in line.split("\t")]
        tokens.append(values[1])
        try:
          token_ids.append(token_to_index[values[1]])
        except KeyError:
          token_ids.append(1)
        if values[4] in ["Postp", "PcAbl", "Adv"]:
          pos.append(replace_dict[values[4]])
        else:
          pos.append(values[4])
        heads.append(values[6])
        dep_labels.append(values[7])
      yield key, {
        "sent_id": sent_id,
        "tokens": tokens,
        "token_ids": token_ids,
        "pos": pos,
        "heads": heads,
        "dep_labels": dep_labels,
      }

if __name__ == "__main__":
  sentence_list = read_conllx(os.path.join(_DATA_DIR, _DEV_FILE))
  converted = convert_to_dict(sentence_list)
  for sentence in converted:
    print(sentence)
    input()

