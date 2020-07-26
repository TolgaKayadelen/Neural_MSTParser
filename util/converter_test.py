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
    test_data_path = os.path.join(_TESTDATA_DIR, "converter")
    propbank_conll = os.path.join(test_data_path, "propbank_ud_testdata_conll.txt")
    converter = cvt.PropbankConverter(propbank_conll)
    converted_sentences = []
    conll_df_list = converter._ReadCoNLLDataFrame(converter._corpus)
    for df in conll_df_list:
      df = df.dropna(how='all').reset_index(drop=True)
      cols = [0, 6]
      df = df.astype({c: int for c in cols})
      if not df.empty:
        sentence_proto = converter._DataFrameToProto(df)
        converted_sentences.append(sentence_proto)
    expected_treebank = reader.ReadTreebankTextProto(
        os.path.join(test_data_path, "propbank_ud_testdata_proto.pbtxt"))
    expected_sentences = expected_treebank.sentence
    for converted, expected in zip(converted_sentences, expected_sentences):
      self.assertEqual(converted, expected)
    print("Passed!")
  
  def test_ConvertDependencyTreebank(self):
    print("Running test_ConvertDependencyTreebank")
    test_data_path = os.path.join(_TESTDATA_DIR, "converter")
    depbank_conll = os.path.join(test_data_path, "depbank_ud_testdata_conll.txt")
    converter = cvt.PropbankConverter(depbank_conll)
    converted_sentences = converter.ConvertConllToProto(converter.sentence_list)
    expected_treebank = reader.ReadTreebankTextProto(
        os.path.join(test_data_path, "depbank_ud_testdata_proto.pbtxt"))
    expected_sentences = expected_treebank.sentence
    for converted, expected in zip(converted_sentences, expected_sentences):
      self.assertEqual(converted, expected)
    print("Passed!")
  
if __name__ == "__main__":
  unittest.main()