# -*- coding: utf-8 -*-

import csv
from datetime import datetime
import os
from google.protobuf import text_format

def write_file(file_content, path):
    """Writes file content to the path and creates all intermediate directories.
    
    Raises:
        OSError: path does not specify a valid path to a file. 

    Args:   
        file_content: string, bythe string content of the file which will be written to 
            the path.
        path: string, path to the file to which the file content will be written. 
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory) # might throw OSError
    
    with open(path, "w") as output_file:
        output_file.write(file_content)
    
def write_proto_as_text(message, path):
    """Writes protocol buffer message to the path in pbtxt format. 
    
    Raises:
        OSError: path does not specify a path to a file.
    
    Args:
        message: a protocol buffer message.
        path: string, path to the file. 
    """
    with open(path, "w") as output_file:
        text_format.PrintMessage(message, output_file, as_utf8=True)

def write_protolist_as_text(messages, path):
  """Writes a list of protos to the path in the pbtxt format.
  
  Sepearetely from the above method, this method is useful to write a 
  list of messages of the same type to a single proto file.
  """
  with open(path, "w") as output_file:
    for message in messages:
      text_format.PrintMessage(message, output_file, as_utf8=True)
  
def write_proto_as_proto(message, path):
    """Serializes a protocol buffer message.
    
    Args:
        message: a protocol buffer message
        path: string, path to the output file
    """
    write_file(message.SerializeToString(), path)
    
def write_model_output(model_dict, parser=False, labeler=False):
  """Writes the model output to a tsv file.
  
  Args:
    model_output_dict: a dict containing data about model training and eval.
    parser: Whether to write output for the dependency parser.
    labeler: Whether to write output for the dependency labeler.
  """
  MODEL_EXP_DIR = "model/experiments"
  assert labeler or parser, "Either labeler or parser should be set to True!!"
  if labeler:
    f_ = "labeler_exp.tsv"
  else:
    f_ = "parser_exp.tsv"
  file_ = os.path.join(MODEL_EXP_DIR, f_)
  file_exists = os.path.isfile(file_)
  with open(os.path.join(file_), "a") as tsvfile:
    fieldnames = ["time", "train_data", "train_data_size", "test_data", "test_data_size",
    "train_acc", "test_acc_unavg", "test_acc_avg", "epochs", "learning_rate", "features", "feature_count"]
    writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
    if not file_exists:
      writer.writeheader()
    writer.writerow({
      "time": datetime.now().strftime("%D, %H:%M:%S"),
      "train_data": model_dict["train_data"],
      "train_data_size": model_dict["train_data_size"],
      "test_data": model_dict["test_data"],
      "test_data_size": model_dict["test_data_size"],
      "train_acc": model_dict["train_acc"],
      "test_acc_unavg": model_dict["test_acc_unavg"],
      "test_acc_avg": model_dict["test_acc_avg"],
      "epochs": model_dict["epochs"],
      "learning_rate": model_dict["learning_rate"],
      "features": model_dict["features"],
      "feature_count": model_dict["feature_count"]}
      )
      