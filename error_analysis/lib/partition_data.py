import os
import collections
import csv
from collections import OrderedDict
from data.treebank import treebank_pb2
from data.treebank import sentence_pb2
from util import reader, writer, common

def is_projective(sentence):
  """Determines whether a sentence is projective.

  Projectivity means that a head can reach all its dependents via other dependents.
  """
  for token in sentence.token[1:]:
    deps = [tok for tok in sentence.token if tok.selected_head.address == token.index]
    for dep in deps:
      btw_tokens = common.GetBetweenTokens(sentence, token, dep)
      if not btw_tokens:
        continue
      for btw_token in btw_tokens:
        if _can_reach(sentence, btw_token, head=token):
          continue
        else:
          return False
  return True


def _can_reach(sentence, dependent_token, head):
  if dependent_token.selected_head.address == head.index:
    return True
  next_head = [tok for tok in sentence.token if tok.index == dependent_token.selected_head.address][0]
  if next_head.word == "ROOT":
    return False
  # Recursive call to _can_ready with the next head.
  return _can_reach(sentence, next_head, head)

class Partition:
  def __init__(self, gold_treebank, test_treebank, language):
    self.gold = reader.ReadTreebankTextProto(gold_treebank)
    self.test = reader.ReadTreebankTextProto(test_treebank)
    self.gold_sentences = {sentence.sent_id: sentence for sentence in self.gold.sentence}
    self.test_sentences = {sentence.sent_id: sentence for sentence in self.test.sentence}
    self.language = language

  @staticmethod
  def write(treebank, lang_dir: str, subdir: str, filename: str):
    writer.write_proto_as_text(treebank, os.path.join(os.path.join(lang_dir, subdir), filename))

  def by_sentence_length(self):
    less_ten_test, less_ten_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    less_twenty_test, less_twenty_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    less_thirty_test, less_thirty_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    less_forty_test, less_forty_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    less_fifty_test, less_fifty_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    over_fifty_test, over_fifty_gold = treebank_pb2.Treebank(), treebank_pb2.Treebank()
    for sent_id, sentence in self.gold_sentences.items():
      sentence_gold = sentence
      sentence_test = self.test_sentences[sent_id]
      print(sentence_gold.sent_id, sentence_test.sent_id)
      assert sentence_gold.sent_id == sentence_test.sent_id, "Mismatching sentences"
      if len(sentence.token) < 10:
        sentence = less_ten_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = less_ten_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
      elif len(sentence.token) < 20:
        sentence = less_twenty_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = less_twenty_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
      elif len(sentence.token) < 30:
        sentence = less_thirty_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = less_thirty_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
      elif len(sentence.token) < 40:
        sentence = less_forty_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = less_forty_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
      elif len(sentence.token) < 50:
        sentence = less_fifty_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = less_fifty_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
      else:
        sentence = over_fifty_test.sentence.add()
        sentence.CopyFrom(sentence_test)
        sentence = over_fifty_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)

    lang_dir = f"./error_analysis/{self.language}-pud/sentence_length"
    writer.write_proto_as_text(less_ten_test, os.path.join(os.path.join(lang_dir, "10"), "test.pbtxt"))
    writer.write_proto_as_text(less_ten_gold, os.path.join(os.path.join(lang_dir, "10"), "gold.pbtxt"))

    writer.write_proto_as_text(less_twenty_test, os.path.join(os.path.join(lang_dir, "20"), "test.pbtxt"))
    writer.write_proto_as_text(less_twenty_gold, os.path.join(os.path.join(lang_dir, "20"), "gold.pbtxt"))

    writer.write_proto_as_text(less_thirty_test, os.path.join(os.path.join(lang_dir, "30"), "test.pbtxt"))
    writer.write_proto_as_text(less_thirty_gold, os.path.join(os.path.join(lang_dir, "30"), "gold.pbtxt"))

    writer.write_proto_as_text(less_forty_test, os.path.join(os.path.join(lang_dir, "40"), "test.pbtxt"))
    writer.write_proto_as_text(less_forty_gold, os.path.join(os.path.join(lang_dir, "40"), "gold.pbtxt"))

    writer.write_proto_as_text(less_fifty_test, os.path.join(os.path.join(lang_dir, "50"), "test.pbtxt"))
    writer.write_proto_as_text(less_fifty_gold, os.path.join(os.path.join(lang_dir, "50"), "gold.pbtxt"))

    writer.write_proto_as_text(over_fifty_test, os.path.join(os.path.join(lang_dir, "over50"), "test.pbtxt"))
    writer.write_proto_as_text(over_fifty_gold, os.path.join(os.path.join(lang_dir, "over50"), "gold.pbtxt"))

  def by_projectivity(self):
    projective_gold = treebank_pb2.Treebank()
    projective_test = treebank_pb2.Treebank()
    non_proj_gold = treebank_pb2.Treebank()
    non_proj_test = treebank_pb2.Treebank()
    for sent_id, sentence_gold in self.gold_sentences.items():
      sentence_test = self.test_sentences[sent_id]
      print(sentence_gold.sent_id, sentence_test.sent_id)
      assert sentence_gold.sent_id == sentence_test.sent_id, "Mismatching sentences"
      if self.is_projective(sentence_gold):
        sentence = projective_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
        sentence = projective_test.sentence.add()
        sentence.CopyFrom(sentence_test)
      else:
        sentence = non_proj_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
        sentence = non_proj_test.sentence.add()
        sentence.CopyFrom(sentence_test)
    lang_dir = f"./error_analysis/{self.language}-pud/projectivity"
    writer.write_proto_as_text(projective_test, os.path.join(os.path.join(lang_dir, "projective"), "test.pbtxt"))
    writer.write_proto_as_text(projective_gold, os.path.join(os.path.join(lang_dir, "projective"), "gold.pbtxt"))
    writer.write_proto_as_text(non_proj_test, os.path.join(os.path.join(lang_dir, "non-projective"), "test.pbtxt"))
    writer.write_proto_as_text(non_proj_gold, os.path.join(os.path.join(lang_dir, "non-projective"), "gold.pbtxt"))

  def by_xcomp(self):
    control_gold = treebank_pb2.Treebank()
    control_test = treebank_pb2.Treebank()
    for sent_id, sentence_gold in self.gold_sentences:
      sentence_test = self.test_sentences[sent_id]
      print(sentence_gold.sent_id, sentence_test.sent_id)
      assert sentence_gold.sent_id == sentence_test.sent_id, "Mismatching sentences"
      labels = [token.label for token in sentence_gold.token]
      if "xcomp" in labels:
        sentence = control_gold.sentence.add()
        sentence.CopyFrom(sentence_gold)
        sentence = control_test.sentence.add()
        sentence.CopyFrom(sentence_test)
    lang_dir = f"./error_analysis/{self.language}-pud/constructions"
    writer.write_proto_as_text(control_test, os.path.join(os.path.join(lang_dir, "control"), "test.pbtxt"))
    writer.write_proto_as_text(control_gold, os.path.join(os.path.join(lang_dir, "control"), "gold.pbtxt"))

  def by_construction(self, construction, data_dir, multiclause=False):
    gold = treebank_pb2.Treebank()
    test = treebank_pb2.Treebank()

    total_sentence_including_const = 0
    total_multiclause_including_const = 0
    tokens_marked_with_const = 0
    sent_ids_with_both_errors = []
    sent_ids_only_label_errors = []
    sent_ids_only_head_errors = []
    sent_ids_only_head_errors_when_multiclause = []
    sent_ids_correct_grammatical_role_attached_to_wrong_pred = []
    sentences_with_const = []
    multiclause_sentences_with_const = []
    multiclause_sentences_with_errors = []
    n_correct_grammatical_role_attached_to_wrong_pred = 0
    errors_for_const = 0
    errors_for_const_in_multiclause = 0
    n_only_head_errors = 0
    n_only_head_errors_when_multiclause = 0
    n_only_label_errors = 0
    n_both_errors = 0
    labels_confused_when_head_correct = collections.Counter()
    labels_confused_when_both_wrong = collections.Counter()
    stats_dict = collections.OrderedDict()
    stats = 0
    for sent_id, sentence_gold in self.gold_sentences.items():
      sentence_test = self.test_sentences[sent_id]
      print(sentence_gold.sent_id, sentence_test.sent_id)
      assert sentence_gold.sent_id == sentence_test.sent_id, "Mismatching sentences"
      labels = [token.label for token in sentence_gold.token]
      if len(sentence_gold.token) < 10:
        sentence_length = 10
      elif len(sentence_gold.token) < 20:
        sentence_length = 20
      elif len(sentence_gold.token) < 30:
        sentence_length = 30
      elif len(sentence_gold.token) < 40:
        sentence_length = 40
      else:
        sentence_length = 50
      if multiclause:
        multiclause_sentence = False
        multiclause_labels = ["csubj", "ccomp", "xcomp"]
        if any(multiclause_label in labels for multiclause_label in multiclause_labels):
          multiclause_sentence = True
        elif "conj" in labels:
          conj_tokens = [token for token in sentence_gold.token if token.label == "conj"]
          # print("conj tokens ", conj_tokens)
          # input()
          for conj_token in conj_tokens:
            if conj_token.category == "VERB":
              multiclause_sentence = True
      if construction in labels:
        sentences_with_const.append(sent_id)
        total_sentence_including_const += 1
        if multiclause_sentence:
          total_multiclause_including_const += 1
          multiclause_sentences_with_const.append(sent_id)
        gold_tokens = [token for token in sentence_gold.token if token.label == construction]
        gold_token_indexes = [token.index for token in gold_tokens]
        tokens_marked_with_const += len(gold_tokens)
        test_tokens = [token for token in sentence_test.token if token.index in gold_token_indexes]
        # print("gold tokens ", gold_tokens)
        # print("test tokens ", test_tokens)
        # input()
        assert (len(gold_tokens) == len(test_tokens)), "Mismatching tokens"
        wrong_head, wrong_grammatical_role, erroneous = False, False, False
        only_label_error, only_head_error, both_head_and_label = False, False, False

        # check whether root is correct.
        root_token_gold = [token for token in sentence_gold.token if token.label == "root"]
        root_token_test = [token for token in sentence_test.token if token.label == "root"]
        # print("gold root ", root_token_gold, "test root ", root_token_test)
        if not root_token_test:
          root_correct = False
        else:
          root_correct = root_token_gold[0].index == root_token_test[0].index
        # print("root correct ", root_correct)
        # input()
        for gold_token, test_token in zip(gold_tokens, test_tokens):
          assert (gold_token.word == test_token.word), "Mismatching tokens"
          csv_line = collections.OrderedDict({"sent_id": None, "length": 0, "multiclause": None, "token": None,  "error": False, "error_type": "none",
                                              "correct_label": None, "confused_label": "none", "pred_arg_error": False,
                                              "multiclause_wrong_pred": False, "root_correct": None, "attached_label": None, "srl1": "none", "srl2": "none", "srl3": "none"})
          stats += 1
          csv_line["sent_id"] = sent_id
          csv_line["length"] = sentence_length
          csv_line["multiclause"] = multiclause_sentence
          csv_line["token"] = gold_token.word
          csv_line["correct_label"] = gold_token.label
          csv_line["root_correct"] = root_correct
          srl_labels = gold_token.srl
          if len(srl_labels) == 1:
            csv_line["srl1"] = srl_labels[0]
          elif len(srl_labels) == 2:
            csv_line["srl1"] = srl_labels[0]
            csv_line["srl2"] = srl_labels[1]
          elif len(srl_labels) > 2:
            csv_line["srl1"] = srl_labels[0]
            csv_line["srl2"] = srl_labels[1]
            csv_line["srl3"] = srl_labels[2]
          if test_token.label != gold_token.label:
            wrong_grammatical_role = True
          if test_token.selected_head.address != gold_token.selected_head.address:
            wrong_head = True
          if wrong_head or wrong_grammatical_role:
            erroneous = True
          if erroneous:
            errors_for_const += 1
            csv_line["error"] = erroneous
            if multiclause_sentence:
              errors_for_const_in_multiclause += 1
              multiclause_sentences_with_errors.append(sent_id)
            if wrong_head and wrong_grammatical_role:
              both_head_and_label = True
              labels_confused_when_both_wrong[f"{gold_token.label}_{test_token.label}"] += 1
              n_both_errors += 1
              csv_line["error_type"] = "BOTH"
              csv_line["confused_label"] = test_token.label
              csv_line["pred_arg_error"] = True
            elif wrong_grammatical_role and not wrong_head:
              only_label_error = True
              n_only_label_errors += 1
              csv_line["error_type"] = "ROLE"
              labels_confused_when_head_correct[f"{gold_token.label}_{test_token.label}"] += 1
              csv_line["correct_label"] = gold_token.label
              csv_line["confused_label"] = test_token.label
              if gold_token.label in ["nsubj", "csubj"] and test_token.label in ["nsubj", "csubj"]:
                csv_line["pred_arg_error"] = False
              elif gold_token.label in ["advcl", "obl"] and test_token.label in ["advcl", "obl"]:
                csv_line["pred_arg_error"] = False
              elif gold_token.label in ["ccomp", "obj", "xcomp"] and test_token.label in ["ccomp", "obj", "xcomp"]:
                csv_line["pred_arg_error"] = False
              else:
                csv_line["pred_arg_error"] = True
            elif wrong_head and not wrong_grammatical_role:
              only_head_error = True
              n_only_head_errors += 1
              csv_line["error_type"] = "ATTACHMENT"
              csv_line["pred_arg_error"] = True
              attached_token = [token for token in sentence_test.token if token.index == test_token.selected_head.address]
              # print("attached token ", attached_token)
              assert(len(attached_token) == 1)
              csv_line["attached_label"] = attached_token[0].label
              # input()
              if multiclause_sentence:
                n_only_head_errors_when_multiclause += 1
                test_token_head = [token for token in sentence_test.token if token.index == test_token.selected_head.address]
                # print('test token head ', test_token_head)
                assert len(test_token_head) == 1, "Can't have more than one head"
                # input()
                if (test_token_head[0].category == "VERB" or test_token_head[0].label == "conj") and test_token.selected_head.address != gold_token.selected_head.address:
                  sent_ids_correct_grammatical_role_attached_to_wrong_pred.append(sent_id)
                  n_correct_grammatical_role_attached_to_wrong_pred += 1
                  csv_line["multiclause_wrong_pred"] = True
          else:
            csv_line["error"] = False
          stats_dict[stats] = csv_line

          if both_head_and_label:
            sent_ids_with_both_errors.append(sent_id)
          if only_label_error:
            sent_ids_only_label_errors.append(sent_id)
          if only_head_error:
            sent_ids_only_head_errors.append(sent_id)
            if multiclause_sentence:
              sent_ids_only_head_errors_when_multiclause.append(sent_id)
          sentence = gold.sentence.add()
          sentence.CopyFrom(sentence_gold)
          sentence = test.sentence.add()
          sentence.CopyFrom(sentence_test)

    with open(os.path.join(os.path.join(data_dir, construction), f"{construction}_errors.csv"), "w") as f:
      csv_writer = csv.DictWriter(f, fieldnames=csv_line.keys())
      csv_writer.writeheader()
      for k, v in stats_dict.items():
        csv_writer.writerow(v)

    # writer.write_proto_as_text(test, os.path.join(os.path.join(data_dir, construction), "test.pbtxt"))
    # writer.write_proto_as_text(gold, os.path.join(os.path.join(data_dir, construction), "gold.pbtxt"))

  def by_dependency_distance(self):
    distance = collections.Counter()
    accurate = collections.Counter()
    total_tokens = 0
    for sent_id, sentence_gold in self.gold_sentences.items():
      sentence_test = self.test_sentences[sent_id]
      print(sentence_gold.sent_id, sentence_test.sent_id)
      assert sentence_gold.sent_id == sentence_test.sent_id, "Mismatching sentences"
      for token_gold, token_test in zip(sentence_gold.token[1:], sentence_test.token[1:]):
        d = abs(token_gold.selected_head.address - token_gold.index)
        if d > 11:
          d = 11
        distance[d] += 1
        if token_test.selected_head.address == token_gold.selected_head.address:
          accurate[d] += 1
    print(distance)
    print(accurate)
    for key in accurate.keys():
      print(accurate[key] / distance[key])


if __name__ == "__main__":
  partition = Partition(
    gold_treebank = "./error_analysis/en-pud/gold_test_treebank.pbtxt",
    test_treebank = "./error_analysis/en-pud/parsed_and_labeled_test_treebank.pbtxt",
    language = "en"
  )
  for tag in ["nsubj", "obj", "obl", "iobj", "ccomp", "csubj", "advcl",  "xcomp"]:
  # for tag in ["nsubj"]:
    partition.by_construction(tag, data_dir=f"./error_analysis/en-pud/argstr", multiclause=True)
  # partition = Partition(
  #  gold_treebank = "./error_analysis/tr-pud/gold_test_treebank.pbtxt",
  #  test_treebank = "./error_analysis/tr-pud/parsed_and_labeled_test_treebank.pbtxt",
  #  language = "tr"
  #)
  #partition.by_sentence_length()