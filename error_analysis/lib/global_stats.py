import os
import collections
import csv
from collections import OrderedDict
from data.treebank import treebank_pb2
from data.treebank import sentence_pb2
from util import reader, writer, common

def main(data_dir, gold_treebank, test_treebank):
  gold = reader.ReadTreebankTextProto(os.path.join(data_dir, gold_treebank))
  test = reader.ReadTreebankTextProto(os.path.join(data_dir, test_treebank))
  gold_sentences = {sentence.sent_id: sentence for sentence in gold.sentence}
  test_sentences = {sentence.sent_id: sentence for sentence in test.sentence}
  labels = collections.Counter()
  attachment = collections.Counter()
  role = collections.Counter()
  both = collections.Counter()
  total_errors = collections.Counter()
  no_error = collections.Counter()

  # get the number of all labels
  for sentence in gold.sentence:
    for token in sentence.token:
      labels[token.label] += 1
  print("labels ", labels)

  for sent_id, gold_sentence in gold_sentences.items():
    test_sentence = test_sentences[sent_id]
    print(gold_sentence.sent_id, test_sentence.sent_id)
    assert gold_sentence.sent_id == test_sentence.sent_id, "Mismatching sentences"

    for gold_token, test_token in zip(gold_sentence.token[1:], test_sentence.token[1:]):
      assert (gold_token.word == test_token.word), "Mismatching tokens"
      if test_token.label != gold_token.label and test_token.selected_head.address != gold_token.selected_head.address:
        both[gold_token.label] += 1
        total_errors[gold_token.label] += 1
      elif test_token.label != gold_token.label:
        role[gold_token.label] += 1
        total_errors[gold_token.label] += 1
      elif test_token.selected_head.address != gold_token.selected_head.address:
        attachment[gold_token.label] += 1
        total_errors[gold_token.label] += 1
      else:
        no_error[gold_token.label] += 1

  total_attachment_errors = sum([attachment[key] for key in attachment.keys()])
  total_role_errors = sum([role[key] for key in role.keys()])
  total_both_errors = sum([both[key] for key in both.keys()])
  print("total attachment ", total_attachment_errors)
  print("total role ", total_role_errors)
  print("total both ", total_both_errors)
  sum_errors = sum([total_errors[key] for key in total_errors.keys()])
  print("sum errors ", sum_errors)
  print("attachment ", attachment)
  print("role ", role)
  print("both ", both)
  print("total errors ", total_errors)
  print("no error ", no_error)
  stats_dict = collections.OrderedDict()
  index = 0
  for label in labels.keys():
    index += 1
    csv_line = collections.OrderedDict({"label": None, "count": 0, "errors": 0, "ratio_all_errors": 0, "ratio_role_errors": 0,
                                        "ratio_attachment_errors": 0, "ratio_both_errors": 0})
    csv_line["label"] = label
    csv_line["count"] = labels[label]
    try:
      csv_line["errors"] = total_errors[label]
    except:
      continue
    try:
      total_errors_for_label = total_errors[label] / sum_errors
      print(f"total errors for {label} ", total_errors_for_label)
      csv_line["ratio_all_errors"] = round(total_errors_for_label, 3)
    except KeyError:
      continue
    try:
      attachment_errors_for_label = attachment[label] / sum_errors
      print(f"attachment errors for {label} ", attachment_errors_for_label)
      csv_line["ratio_attachment_errors"] = round(attachment_errors_for_label, 3)
    except KeyError:
      continue
    try:
      role_errors_for_label = role[label] / sum_errors
      print(f"role errors for {label} ", role_errors_for_label)
      csv_line["ratio_role_errors"] = round(role_errors_for_label, 3)
    except KeyError:
      continue
    try:
      both_errors_for_label = both[label] / sum_errors
      print(f"both errors for {label} ", both_errors_for_label)
      csv_line["ratio_both_errors"] = round(both_errors_for_label, 3)
    except KeyError:
      continue
    print("-----")
    stats_dict[index] = csv_line
  print(stats_dict)

  with open(os.path.join(data_dir, "global_stats.csv"), "w") as f:
    csv_writer = csv.DictWriter(f, fieldnames=csv_line.keys())
    csv_writer.writeheader()
    for k, v in stats_dict.items():
      csv_writer.writerow(v)

if __name__ == "__main__":
  data_dir = "./error_analysis/tr-propbank/dev"
  gold_treebank = "tr_without_srl_gold.pbtxt"
  test_treebank = "tr_without_srl_parsed.pbtxt"
  main(data_dir, gold_treebank=gold_treebank, test_treebank=test_treebank)
