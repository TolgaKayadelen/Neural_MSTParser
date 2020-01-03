# -*- coding: utf-8 -*-

"""This module converts xml formatted propbank frames to frame.proto"""

import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join

indent = '\t'

def get_argument_roles(arguments):
  for argument in arguments:
    if argument.tag == "note":
      continue
    attr = argument.attrib
    #print(attr)
    print("{}argument".format(indent*2))
    print("{}number: {}".format(indent*3, attr["n"]))
    print("{}suffix: {}".format(indent*3, attr["suffix"]))
    print("{}description: {}".format(indent*3, attr["descr"].encode("utf-8")))
    vnrole =  argument.findall("vnrole")
    if vnrole:
      if (len(vnrole)) > 1:
        print("this argument has more than one theta roles".format())
        raw_input("press to continue")
      print("{}theta_role: {}".format(indent*3, vnrole[0].attrib["vntheta"])) 


def get_frame(roleset):
  print("{}id: {}".format(indent*2, roleset.attrib["id"].encode("utf-8")))
  print("{}sense: {}".format(indent*2, roleset.attrib["name"].encode("utf-8")))
  arguments = [child for child in roleset if child.tag == "roles"]
  map(get_argument_roles, arguments)


def main():
  #files = ["gel.xml"]
  unique_senses = 0
  path = "./data/propbank/frames"
  files = [f for f in listdir(path) if isfile(join(path, f))]
  #myfiles = files[:50]
  for file_ in files:
    xmlfile = join(path, file_)
    if xmlfile in ["./data/propbank/frames/framesetTR.dtd", "./data/propbank/frames/frameset.dtd"]:
      continue
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    predicates = [child for child in root if child.tag == "predicate"]
    for predicate in predicates:
      print("PREDICATE")
      print("{}lemma: {}".format(indent, predicate.attrib["lemma"].encode("utf-8")))
      rolesets = [child for child in predicate if child.tag == "roleset"]
      for roleset in rolesets:
        unique_senses += 1
        get_frame(roleset)
  print("total verb senses: {}".format(unique_senses))


if __name__ == "__main__":
  main()
