# -*- coding: utf-8 -*-

"""This module converts xml formatted propbank frames to frame.proto"""

import sys
import itertools
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from util import writer

from data.propbank.protos import frame_pb2
from google.protobuf import text_format

_INPUT_FRAMES_DIR = "./data/propbank/frames_xml"
_OUTPUT_FRAMES_DIR = "./data/propbank/frames_proto"

def get_argnum_enum(n):
  if n in ["0", "1", "2", "3", "4", "5", "6"]:
    argnum = frame_pb2.ArgNum = int(n) + 1
  elif n == "m":
    argnum = frame_pb2.A_LVB
  else:
    argnum = frame_pb2.UNKNOWN_ARG
  return argnum

def get_suffix_enum(suffix):
  if suffix == "NOM":
    return frame_pb2.NOM
  elif suffix == "ACC":
    return frame_pb2.ACC
  elif suffix == "DAT":
    return frame_pb2.DAT
  elif suffix == "LOC":
    return frame_pb2.LOC
  elif suffix == "ABL":
    return frame_pb2.ABL
  elif suffix == "INS":
    return frame_pb2.INS
  elif suffix == "WITH":
    return frame_pb2.INS
  elif suffix == "NONE" or suffix == "":
    return frame_pb2.UNKNOWN_CASE
  else:
    sys.exit("Couldn't find a value for this suffix in the proto {}".format(suffix))

def populate_argument_structure(roleset, arg_str):
  arg_str.id = roleset.attrib["id"]
  arg_str.sense = roleset.attrib["name"]  
  def populate_argument_roles(roles):
    for role in roles:
      if role.tag == "note":
        continue
      argument = arg_str.argument.add()
      argument.description = role.attrib["descr"]
      argument.number = get_argnum_enum(role.attrib["n"])
      suffix = role.attrib["suffix"].split("-")
      argument.case.extend(map(get_suffix_enum, suffix))
      vnrole = role.findall("vnrole")
      if vnrole:
        if len(vnrole) > 1:
          print("This argument has more than one theta roles, taking the first one")
          raw_input("Press to continue")
        argument.theta_role = vnrole[0].attrib["vntheta"]
  
  roles = [child for child in roleset if child.tag == "roles"]
  examples = [child for child in roleset if child.tag == "example"]
  for example in examples:
    for child in example:
      if child.tag == "text":
        arg_str.example = child.text if child.text else "None"
        break
  map(populate_argument_roles, roles)
  return arg_str

def xml_to_proto(files):
  frames = []
  unique_senses = 0
  for file_ in files:
    if file_ in ["framesetTR.dtd", "frameset.dtd"]:
      continue
    print(file_)
    xmlfile = join(_INPUT_FRAMES_DIR, file_)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    frame = frame_pb2.Frame()
    frame.frame_id = file_.split(".")[0]
    predicates = [child for child in root if child.tag == "predicate"]
    for predicate in predicates:
      #verb = frame_pb2.Verb()
      verb = frame.verb.add()
      verb.lemma = predicate.attrib["lemma"]
      rolesets = [child for child in predicate if child.tag == "roleset"]
      for roleset in rolesets:
        unique_senses += 1
        arg_str = verb.argument_structure.add()
        populate_argument_structure(roleset, arg_str)
    frames.append(frame)
  return frames

def main():
  #files = ["gel.xml", "getir.xml"]
  files = [f for f in listdir(_INPUT_FRAMES_DIR) if isfile(join(_INPUT_FRAMES_DIR, f))]
  frames = xml_to_proto(files)
  for frame in frames:
    writer.write_proto_as_text(
      frame, join(_OUTPUT_FRAMES_DIR, "{}.pbtxt".format(frame.frame_id.encode("utf-8"))))
    #print(text_format.MessageToString(verb, as_utf8=True))
    


if __name__ == "__main__":
  main()
