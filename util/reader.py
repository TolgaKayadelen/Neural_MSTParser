# -*- coding: utf-8 -*-

"""Functions to read treebank data in various formats."""

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from proto import metrics_pb2
from google.protobuf import text_format
import argparse
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def ReadConllX(path):
    """Read treebank from a file where sentences are in conll-X format.
    Args:
        corpus: path to the treebank file.
    Returns:
        list of sentences each in Conll-x format.
    """
    from copy import deepcopy
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


def ReadSentenceTextProto(path):
    """Read proto text formatted sentence message from the path.
    Args:
        path: path to the pbtxt file.
    Returns:
        a protocol buffer object.
    """
    file_content = _ReadFile(path)
    return text_format.Parse(file_content, sentence_pb2.Sentence())


def ReadSentenceProto(path):
    """Read proto buffer (binary) formatted sentence message from path.
    Args:
        path: path to binary proto file.
    Returns:
        a protocol buffer object
    """
    sentence = sentence_pb2.Sentence()
    with open(path, "rb") as sentence_proto:
        sentence.ParseFromString(sentence_proto.read())
    return sentence

def ReadTreebankProto(path):
    """Read proto buffer (binary) formatted treebank message from path.
    Args:
        path: path to binary proto file.
    Returns:
        a protocol buffer object
    """
    treebank = treebank_pb2.Treebank()
    with open(path, "rb") as trb_proto:
        treebank.ParseFromString(trb_proto.read())
    return treebank

def ReadTreebankTextProto(path):
    """Read proto text formatted treebank message from the path.
    Args:
        path: path to the pbtxt file.
    Returns:
        a protocol buffer object.
    """
    file_content = _ReadFile(path)
    return text_format.Parse(file_content, treebank_pb2.Treebank())
    
def ReadMetricsTextProto(path):
  """Read proto text formatted metrics message from the path.
  Args:
    path: path to the pbtxt file.
  Returns:
    a protocol buffer object.
  """
  file_content = _ReadFile(path)
  return text_format.Parse(file_content, metrics_pb2.Metrics())

def _ReadFile(path):
    """Reads the file content from the path and returns it as a byte string. 
    
    Args:   
        path: string, path to a file. 
    Raises:
        IOError: file cannot be read from path.
    """
    import codecs
    with codecs.open(path, encoding="utf-8") as in_file:
        read = in_file.read().strip()
    return read


def main(args):
    sentence_list = ReadConllX(args.input_file)
    for line in sentence_list[2]:
        print(line)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to input file.")
    args = parser.parse_args()
    main(args)
 