# -*- coding: utf-8 -*-

"""This module converts xml formatted propbank frames to frame.proto"""

import xml.etree.ElementTree as ET


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
    vnrole,  = [child for child in argument if child.tag == "vnrole"]
    #print(vnrole.attrib)
    print("{}theta_role: {}".format(indent*3, vnrole.attrib["vntheta"])) 


def get_frame(roleset):
  print("{}id: {}".format(indent*2, roleset.attrib["id"].encode("utf-8")))
  print("{}sense: {}".format(indent*2, roleset.attrib["name"].encode("utf-8")))
  arguments = [child for child in roleset if child.tag == "roles"]
  map(get_argument_roles, arguments)


def main():
  xmlfile = "./data/propbank/frames/gönder.xml"
  tree = ET.parse(xmlfile)
  root = tree.getroot()
  predicates = [child for child in root if child.tag == "predicate"]
  for predicate in predicates:
    print("PREDICATE")
    print("{}lemma: {}".format(indent, predicate.attrib["lemma"].encode("utf-8")))
    rolesets = [child for child in predicate if child.tag == "roleset"]
    for roleset in rolesets:
      get_frame(roleset)


if __name__ == "__main__":
  main()
