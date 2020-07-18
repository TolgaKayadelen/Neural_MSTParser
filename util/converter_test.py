"""Unit tests for converter module"""

import os
import unittest

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from google.protobuf import text_format
from util import reader
from util import converter as cvt

_TESTDATA_DIR = "data/testdata"

class ConverterTest(unittest.TestCase):
  def test_ConvertPropbank(self):
    print("Running test_ConvertPropbank")
    propbank_path = os.path.join(_TESTDATA_DIR, "propbank")
    propbank_conll = os.path.join(propbank_path, "propbank_ud_testdata_conll.txt")
    converter = cvt.Converter(propbank_conll)
    sentences = converter.sentence_list
    converted_sentences = converter.ConvertConllToProto(sentences)
    expected_treebank = reader.ReadTreebankTextProto(
        os.path.join(propbank_path, "propbank_ud_testdata_proto.pbtxt"))
    expected_sentences = expected_treebank.sentence
    for converted, expected in zip(converted_sentences, expected_sentences):
      self.assertEqual(converted, expected)
    print("Passed!")
  
if __name__ == "__main__":
  unittest.main()