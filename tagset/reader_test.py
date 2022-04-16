# -*- coding: utf-8 -*-

import os

from absl.testing import parameterized
from absl.testing import absltest
from google.protobuf import text_format
from tagset.reader import LabelReader

# import logging
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_POS_CATEGORIES = {'ADJ': 1,
                   'ADP': 2,
                   'ADV': 3,
                   'AUX': 4,
                   'CCONJ': 5,
                   'DET': 6,
                   'INTJ': 7,
                   'NOUN': 8,
                   'PART': 9,
                   'NUM': 10,
                   'PRON': 11,
                   'PROPN': 12,
                   'PUNCT': 13,
                   'SCONJ': 14,
                   'SYM': 15,
                   'VERB': 16,
                   'X': 17,
                   'TOP': 18,
                   '-pad-': 0
                  }

_POS_TAGS = {'ANum': 1,
             'Abr': 2,
             'Adj': 3,
             'Adverb': 4,
             'Conj': 5,
             'Demons': 6,
             'Det': 7,
             'Dup': 8,
             'Interj': 9,
             'NAdj': 10,
             'NNum': 11,
             'Neg': 12,
             'Ness': 13,
             'Noun': 14,
             'PCAbl': 15,
             'PCAcc': 16,
             'PCDat': 17,
             'PCGen': 18,
             'PCIns': 19,
             'PCNom': 20,
             'Pers': 21,
             'PostP': 22,
             'Prop': 23,
             'Punc': 24,
             'Quant': 25,
             'Ques': 26,
             'Reflex': 27,
             'Rel': 28,
             'Since': 29,
             'TOP': 30,
             'Verb': 31,
             'With': 32,
             'Without': 33,
             'Zero': 34,
             '-pad-': 0
            }

_DEP_LABELS = {'acl': 1,
               'advcl': 2,
               'advmod': 3,
               'advmod:emph': 4,
               'amod': 5,
               'appos': 6,
               'aux': 7,
               'aux:q': 8,
               'case': 9,
               'cc': 10,
               'cc:preconj': 11,
               'ccomp': 12,
               'clf': 13,
               'compound': 14,
               'compound:lvc': 15,
               'compound:redup': 16,
               'conj': 17,
               'cop': 18,
               'csubj': 19,
               'dep': 20,
               'det': 21,
               'discourse': 22,
               'dislocated': 23,
               'fixed': 24,
               'flat': 25,
               'goeswith': 26,
               'iobj': 27,
               'list': 28,
               'mark': 29,
               'nmod': 30,
               'nmod:poss': 31,
               'nsubj': 32,
               'nummod': 33,
               'obj': 34,
               'obl': 35,
               'orphan': 36,
               'parataxis': 37,
               'punct': 38,
               'root': 39,
               'vocative': 40,
               'xcomp': 41,
               'TOP': 42,
               '-pad-': 0
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


_MORPH = {'UNKNOWN_MORPH': 0,
          'abbr_yes': 1,
          'aspect_hab': 2,
          'aspect_imp': 3,
          'aspect_perf': 4,
          'aspect_prog': 5,
          'case_abl': 6,
          'case_acc': 7,
          'case_dat': 8,
          'case_equ': 9,
          'case_gen': 10,
          'case_ins': 11,
          'case_loc': 12,
          'case_nom': 13,
          'echo_rdp': 14,
          'evident_fh': 15,
          'evident_nfh': 16,
          'mood_cnd': 17,
          'mood_des': 18,
          'mood_gen': 19,
          'mood_imp': 20,
          'mood_ind': 21,
          'mood_nec': 22,
          'mood_opt': 23,
          'mood_pot': 24,
          'number_psor_plur': 25,
          'number_psor_sing': 26,
          'number_plur': 27,
          'number_sing': 28,
          'numtype_card': 29,
          'numtype_dist': 30,
          'numtype_ord': 31,
          'person_psor_1': 32,
          'person_psor_2': 33,
          'person_psor_3': 34,
          'person_1': 35,
          'person_2': 36,
          'person_3': 37,
          'polarity_neg': 38,
          'polarity_pos': 39,
          'polite_infm': 40,
          'prontype_dem': 41,
          'prontype_ind': 42,
          'prontype_prs': 43,
          'reflex_yes': 44,
          'tense_fut': 45,
          'tense_past': 46,
          'tense_pqp': 47,
          'tense_pres': 48,
          'verbform_conv': 49,
          'verbform_part': 50,
          'verbform_vnoun': 51,
          'voice_cau': 52,
          'voice_pass': 53,
          'voice_rcp': 54,
          'voice_rfl': 55,
          '-pad-': 0}

class LabelReaderTest(parameterized.TestCase):
  """Tests for the label reader."""
  @parameterized.named_parameters(
    [
      {
        "testcase_name": "pos",
        "tagset": "pos",
        "reverse": False,
        "expected_labels": _POS_TAGS,
      },
      {
        "testcase_name": "category",
        "tagset": "category",
        "reverse": False,
        "expected_labels": _POS_CATEGORIES,
      },
      {
        "testcase_name": "category_reversed",
        "tagset": "category",
        "reverse": True,
        "expected_labels": {k:v for v,k in _POS_CATEGORIES.items()}
      },
      {
        "testcase_name": "dep_labels",
        "tagset": "dep_labels",
        "reverse": False,
        "expected_labels": _DEP_LABELS,
      },
      {
        "testcase_name": "semantic_roles",
        "tagset": "srl",
        "reverse": False,
        "expected_labels": _SEMANTIC_ROLES,
      },
      {
        "testcase_name": "morphology",
        "tagset": "morph",
        "reverse": False,
        "expected_labels": _MORPH,
      }
    ]
  )
  def test_read_labels(self, tagset, reverse, expected_labels):
    self.assertDictEqual(LabelReader.get_labels(tagset, reverse).labels,
                         expected_labels)
    # if reverse:
    #   print(LabelReader.get_labels(tagset, reverse).labels)
    print("Passed!")
  
  def test_itov(self):
    label_reader = LabelReader.get_labels("dep_labels")
    self.assertEqual(label_reader.itov(10), "cc")
    print("Passed!")
  
  def test_vtoi(self):
    label_reader = LabelReader.get_labels("pos")
    self.assertEqual(label_reader.vtoi("Adj"), 3)
    print("Passed!")

if __name__ == "__main__":
	absltest.main()