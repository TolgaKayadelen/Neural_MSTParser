from absl.testing import parameterized
from absl.testing import absltest
from collections import OrderedDict
from data.treebank import sentence_pb2
from error_analysis.lib import partition_data


class PartitionData(parameterized.TestCase):
  """Tests for the data partition module."""

  @parameterized.named_parameters(
    [
      {
        "testcase_name": "not_projective",
        "sentence_dict": OrderedDict({"ROOT": 0, "John": 2, "cancelled": 0, "our": 4, "flight": 2,
                                      "this": 6, "morning": 2, "which": 8, "was": 4, "late": 8}),
        "projective": False
      },
      {
        "testcase_name": "projective",
        "sentence_dict": OrderedDict({"ROOT": 0, "John": 2, "is": 0, "a": 4, "smart": 5,
                                      "kid": 2}),
        "projective": True
      }
    ]
  )
  def test_projectivity(self, sentence_dict, projective):
    sentence = sentence_pb2.Sentence()
    idx = 0
    for word, head in sentence_dict.items():
      token = sentence.token.add(word=word, index=idx)
      token.selected_head.address = head
      idx += 1
    is_projective = partition_data.is_projective(sentence)
    self.assertEqual(is_projective, projective)


if __name__ == "__main__":
  absltest.main()