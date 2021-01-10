# -*- coding: utf-8 -*-

import os

from absl.testing import parameterized
from absl.testing import absltest
from google.protobuf import text_format
from tagset.reader import LabelReader

# import logging
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_POS_CATEGORIES = {'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7,
                   'NOUN': 8, 'NUM': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'TOP': 13, 'VERB': 14,
                   'X': 15, '-pad-': 0}

_POS_TAGS = {'ANum': 1, 'Abr': 2, 'Adj': 3, 'Adverb': 4, 'Conj': 5, 'Demons': 6, 'Det': 7,
            'Dup': 8, 'Interj': 9, 'NAdj': 10, 'NNum': 11, 'Neg': 12, 'Ness': 13, 'Noun': 14, 
            'PCAbl': 15, 'PCAcc': 16, 'PCDat': 17, 'PCGen': 18, 'PCIns': 19, 'PCNom': 20, 'Pers': 21, 
            'PostP': 22, 'Prop': 23, 'Punc': 24, 'Quant': 25, 'Ques': 26, 'Reflex': 27, 'Rel': 28,
            'Since': 29, 'TOP': 30, 'Verb': 31, 'With': 32, 'Without': 33, 'Zero': 34, '-pad-': 0
            }

_DEP_LABELS = {'acl': 1, 'advcl': 2, 'advmod': 3, 'advmod:emph': 4, 'amod': 5, 'appos': 6,
               'aux:q': 7, 'case': 8, 'cc': 9, 'ccomp': 10, 'compound': 11, 'compound:lvc': 12,
               'compound:redup': 13, 'conj': 14, 'cop': 15, 'csubj': 16, 'dep': 17, 'det': 18,
               'discourse': 19, 'fixed': 20, 'flat': 21, 'mark': 22, 'nmod': 23, 'nmod:poss': 24,
               'nsubj': 25, 'nummod': 26, 'obj': 27, 'obl': 28, 'parataxis': 29, 'pobj': 30,
               'prep': 31, 'punct': 32, 'root': 33, 'TOP': 34, 'vocative': 35, '-pad-': 0
              }

_SEMANTIC_ROLES = {'-pad-': 0, 'B-A0': 1, 'I-A0': 2, 'B-A1': 3, 'I-A1': 4, 'B-A3': 5, 'I-A3': 6, 
                   'B-A2': 7, 'I-A2': 8, 'B-A4': 9, 'I-A4': 10, 'B-A-A': 11, 'I-A-A': 12,
                   'B-AM-ADV': 13, 'I-AM-ADV': 14, 'B-AM-CAU': 15, 'I-AM-CAU': 16, 'B-AM-COM': 17,
                   'I-AM-COM': 18, 'B-AM-DIR': 19, 'I-AM-DIR': 20, 'B-AM-DIS': 21, 'I-AM-DIS': 22,
                   'B-AM-EXT': 23, 'I-AM-EXT': 24, 'B-AM-GOL': 25, 'I-AM-GOL': 26, 'B-AM-INS': 27,
                   'I-AM-INS': 28, 'B-AM-LOC': 29, 'I-AM-LOC': 30, 'B-AM-LVB': 31, 'I-AM-LVB': 32,
                   'B-AM-MNR': 33, 'I-AM-MNR': 34, 'B-AM-NEG': 35, 'I-AM-NEG': 36, 'B-AM-PRD': 37,
                   'I-AM-PRD': 38, 'B-AM-TMP': 39, 'I-AM-TMP': 40, 'B-AM-TWO': 41, 'I-AM-TWO': 42,
                   'B-C-A0': 43, 'I-C-A0': 44, 'B-C-A1': 45, 'I-C-A1': 46, 'B-C-A2': 47, 'I-C-A2': 48,
                   'B-C-A3': 49, 'I-C-A3': 50, 'B-C-A4': 51, 'I-C-A4': 52, 'B-R-A1': 53, 'I-R-A1': 54,
                   'B-R-A2': 55, 'I-R-A2': 56, 'B-R-A3': 57, 'I-R-A3': 58, 'B-R-A4': 59, 'I-R-A4': 60,
                   'B-AM-MOD': 61, 'I-AM-MOD': 62, 'B-R-A0': 63, 'I-R-A0': 64, 'B-AM-REC': 65,
                   'I-AM-REC': 66, 'B-notset': 67, 'I-notset': 68, 'B-A4-DIR': 69, 'I-A4-DIR': 70,
                   'O': 71, 'V': 72
                  }


class LabelReaderTest(parameterized.TestCase):
  """Tests for the label reader."""
  @parameterized.named_parameters(
    [
      {
        "testcase_name": "pos",
        "tagset": "pos",
        "expected_labels": _POS_TAGS,
      },
      {
        "testcase_name": "category",
        "tagset": "category",
        "expected_labels": _POS_CATEGORIES,
      },
      {
        "testcase_name": "dep_labels",
        "tagset": "dep_labels",
        "expected_labels": _DEP_LABELS,
      },
      {
        "testcase_name": "semantic_roles",
        "tagset": "semantic_roles",
        "expected_labels": _SEMANTIC_ROLES,
      }
    ]
  )
  def test_read_labels(self, tagset, expected_labels):
    self.assertDictEqual(LabelReader.get_labels(tagset).labels, expected_labels)
    print("Passed!")
  

if __name__ == "__main__":
	absltest.main()