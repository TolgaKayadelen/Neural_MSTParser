# -*- coding: utf-8 -*-

"""Tool to apply various data type conversions to treebanks.

Can convert:

Source Format               Target Format
___________________________________________________________
conll-X                     Protocol Buffer
Protocol Buffer             conll-X (not implemented yet!)

Can also keep or remove igs based on the keep_igs parameter.

Usage: 
Convert from conll to protobuf and write as proto
bazel-bin/util/converter 
--input_file=/path/to/input/conll/corpus 
--output_file=/path/to/output_file # do not give file extension
--writetext=True
--writeproto=True

Example:
bazel-bin/util/converter 
--input_file=./data/UDv23/Turkish/UD_Turkish_IMST/tr_imst_ud_dev.conllu 
--output_file=./data/UDv23/Turkish/training/treebank 
--writetext=True


"""
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from util import reader, writer
from util.nn import nn_utils
from util import common
from collections import defaultdict, OrderedDict
from copy import deepcopy
from google.protobuf import text_format

import numpy as np
import argparse
import os
import re
import random
import pandas as pd
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class Converter:
    
    def __init__(self, corpus):
        self._corpus = corpus
        self.sentence_list = reader.ReadConllX(self._corpus)
        
    def ConvertConllToProto(self, conll_sentences, keep_igs=True):
        """Converts conll-X formatted sentences to proto buffer objects.    
        Args:
            conll_sentneces: list of conll sentences.
            output_file: path to file where protobuf sentences will be written.
            writetext: if True, writes output as pbtxt.
            writeproto: if True, writes output as protobuffer binary.
            prototype: sentence proto or treebank proto.
            keep_igs: if False, removes IGs from data.
        Returns:    
            list of proto buffers.
        """
        sentence_protos = []
        logging.debug("Converting sentences to protocol buffer.")
        for conll in conll_sentences:
            sentence = sentence_pb2.Sentence()
            sentence_dict, metadata = self._ConvertToDict(conll)
            
            # add metadata
            if metadata:
              sentence.text = metadata["text"]
              sentence.sent_id = metadata["sent_id"]
            
            # add a root token
            root_token = sentence.token.add()
            root_token.word = "ROOT"
            root_token.category = "TOP"
            root_token.pos = "TOP"
            root_token.label = "TOP"
            root_token.selected_head.address = 0
            root_token.index = 0
            
            # main loop
            for index in sentence_dict:
                token = sentence.token.add()
                token.index = sentence_dict[index]["idx"]
                token.word = sentence_dict[index]["word"]
                token.lemma = sentence_dict[index]["lemma"]
                token.category = sentence_dict[index]["ctag"]
                token.pos = sentence_dict[index]["ftag"]
                if not sentence_dict[index]["feats"] == "_":
                    feats = sentence_dict[index]["feats"].split("|")
                    for feat in feats:
                        morph_feature = token.morphology.add()
                        morph_feature.name = feat.split("=")[0].lower()
                        morph_feature.value = feat.split("=")[1].lower()
                head = token.selected_head
                head.address = sentence_dict[index]["head"]
                head.arc_score = 0.0 # default
                token.label = sentence_dict[index]["rel"]
            
            # add sentence length
            sentence.length = len(sentence.token)
            sentence_protos.append(sentence)
    
        assert len(conll_sentences) == len(sentence_protos)
        logging.info("%d sentences converted to protocol buffer." % len(conll_sentences))
        
        return sentence_protos
        
    def _RemoveIGs(self, sentences):
        #(TODO): need to make sure this works while converting conll to proto.
        # The call from convert to proto to this function is not implemented yet. 
        """Removes inflection groups and creates a word based data. 
        
        PoS Tags, morphological features and dependencies are adjusted accordingly after 
        the igs are removed. 
        
        Args:
            sentences: list of sentences each in conll-X format.
        Returns:
            list of sentences in conll-X format where igs are removed. 
        """
        
        sentences_wo_igs_dict = []
        for sentence in sentences:
            sentence_dict, _ = self._ConvertToDict(sentence)
            ig_tokens = self._FindIgTokens(sentence_dict)
            new_s = deepcopy(sentence_dict)
            for idx in ig_tokens:
                new_s[idx+1]["lemma"] = new_s[idx]["lemma"]
                if new_s[idx+1]["ftag"] == "Zero" \
                or new_s[idx+1]["ftag"] == "_":
                    new_s[idx+1]["ftag"] = new_s[idx+1]["ctag"]
                removed_token = new_s[idx]
                del new_s[idx]
                
                for token in new_s:
                    if new_s[token]["idx"] < removed_token["idx"]:
                        if new_s[token]["head"] == removed_token["idx"]:
                            continue
                        if new_s[token]["head"] > removed_token["idx"]:
                            new_s[token]["head"] -= 1
                    if new_s[token]["idx"] > removed_token["idx"]:
                        new_s[token]["idx"] -= 1
                        if new_s[token]["head"] == 0:
                            continue
                        if not new_s[token]["head"] > removed_token["idx"]:
                            continue
                        new_s[token]["head"] -= 1
            sentences_wo_igs_dict.append(new_s)

        assert len(sentences_wo_igs_dict) == len(sentences)
        #Convert sentences back to conll-X format.
        sentences_wo_igs = self._ConvertToConllX(sentences_wo_igs_dict)    
        return sentences_wo_igs

    
    def _ConvertToDict(self, sentence):
        """Converts a conll-X formatted sentence to a dictionary.
        
        This is an intermediate step needed:
            a) removing igs from a sentence and/or
            b) converting conll sentences to Protocol Buffers.
        
        Args:
            sentence: a sentence in conll-X format.
            semantic_roles: if True, also adds semantic role label to the token.
        Returns:
            sentence mapped to a defaultdict.
            metadata: the text and the id of the sentence.
        """
        sentence_dict = defaultdict(OrderedDict)
        metadata = {}
        token = 0
        for line in sentence:
            if line.startswith('# newdoc'):
                continue
            if line.startswith('# Change'):
                continue
            if line.startswith("# Fixed"):
                continue
            if line.startswith("# Fix"):
                continue
            if line.startswith("# NOTE"):
                continue
            if line.startswith('# Checktree'):
                continue
            if line.startswith('# global'):
              continue
            if line.startswith('# s_type'):
              continue
            if line.startswith("# meta"):
              continue
            if line.startswith("# newpar"):
              continue
            if line.startswith("# speaker"):
              continue
            if line.startswith("# addressee"):
              continue
            if line.startswith("# trailing_xml"):
              continue
            if line.startswith("# TODO"):
              continue
            if line.startswith("# orig"):
              continue
            if line.startswith("# sent_id"):
                metadata["sent_id"] = line.split("=")[1].strip()
                continue
            if line.startswith("# text"):
                metadata["text"] = line.split("=")[1].strip()
                continue
            if re.match("(([0-9]|[1-9][0-9]|[1-9][0-9][0-9])-([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))", line):
                print("skipping {}".format(line))
                continue
            if re.match("([0-9][0-9]\.[0-9]|[0-9]\.[0-9])", line):
              print("skipping {}".format(line))
              # input()
              continue
            values = [item.strip() for item in line.split("\t")]
            sentence_dict[token]["idx"] = int(values[0])
            sentence_dict[token]["word"] = values[1]
            sentence_dict[token]["lemma"] = values[2]
            sentence_dict[token]["ctag"] = values[3]
            sentence_dict[token]["ftag"] = values[4]
            sentence_dict[token]["feats"] = values[5]
            sentence_dict[token]["head"] = int(values[6])
            sentence_dict[token]["rel"] = values[7]
            sentence_dict[token]["deps"] = values[8]
            sentence_dict[token]["misc"] = values[9]
            token += 1
        return sentence_dict, metadata
    
    
    def _ConvertToConllX(self, sentences):
        """Converts dictionary formatted sentences to Conll-X format.
        
        Args: 
            sentences: sentences in dictionary format.     
        Returns:
            sentences_conll = sentences in conll-x format)
        """
        sentences_list = []
        for sentence in sentences:
            sentence_lines = []
            line = []
            for token in sentence:
                for key in sentence[token]:
                    value = sentence[token][key]
                    line.append(str(value))
                sentence_lines.append("\t".join(line)+"\n")
                del line[:]
            sentences_list.append(sentence_lines)
            #del sentence_lines[:]   
        
        #print(sentences_list)
        assert len(sentences_list) == len(sentences) 
        return sentences_list           
                    
                    
    def _FindIgTokens(self, sentence):
        """Find indices of the IG tokens. 
        
        Args:
            sentence: a sentence in dictionary format. 
        Returns:
            list of ig token indices in the sentence.
        """
        ig_indices = []
        for token in sentence:
            if sentence[token]["word"] == "_":
                idx = token
                ig_indices.append(idx)
        return ig_indices


    def Write(self, sentence_protos, output_file, prototype="treebank", writetext=False, writeproto=False):    
        assert prototype in ["sentence", "treebank"], "Unknown prototype!!"
        
        if prototype == "sentence":
            if writetext and output_file:
                text_output = output_file + ".pbtxt"
                for sentence_proto in sentence_protos:
                    writer.write_proto_as_text(sentence_proto, text_output)
            if writeproto and output_file:
                proto_output = output_file + ".protobuf"
                for sentence_proto in sentence_protos:
                    writer.write_proto_as_proto(sentence_proto, proto_output)
        else:
            if writetext and output_file:
                text_output = output_file + ".pbtxt"
                treebank = treebank_pb2.Treebank()
                for sentence_proto in sentence_protos:
                    s = treebank.sentence.add()
                    s.CopyFrom(sentence_proto)
                writer.write_proto_as_text(treebank, text_output)
            if writeproto and output_file:
                proto_output = output_file + ".protobuf"
                treebank = treebank_pb2.Treebank()
                for sentence_proto in sentence_protos:
                    s = treebank.sentence.add()
                    s.CopyFrom(sentence_proto)
                writer.write_proto_as_proto(treebank, proto_output)

        logging.info(f"{len(sentence_protos)} sentences written to {output_file}")    



pd.options.display.max_columns = 100
class PropbankConverter(Converter):
  """A special class for propbank conversion."""
  def __init__(self, corpus):
    super().__init__(corpus)
  
  def _ReadCoNLLDataFrame(self, conll):
    conll_df = pd.read_table(conll, sep="\t", quoting=3, header=None,
                            names=list(range(0,40)),
                            skip_blank_lines=False)
    # TODO: turn this into a function decorator.
    return self._ChunkDataFrame(conll_df)
  
  def _ChunkDataFrame(self, df):
    """Converts one big DF which has all sentenes to a list of DFs."""
    df_list = np.split(df, df[df.isnull().all(1)].index)
    return df_list
  
  def _DataFrameToProto(self, df, semantic_roles=True):
    sentence = sentence_pb2.Sentence()
    if semantic_roles:
      predicates = {}
      predicate_counter = 0
    # first we add root token
    sentence.token.add(
      word="ROOT",
      category="TOP",
      pos="TOP",
      label="TOP",
      selected_head=sentence_pb2.Head(address=0),
      index=0
    )
    # print(df)
    # input("...")
    # then we add all tokens with syntactic information to the sentence.
    for idx, row in df.iterrows():
      token = sentence.token.add(
        index=row[0],
        word=row[1],
        lemma=row[2],
        category=row[3],
        pos=row[4],
        selected_head=sentence_pb2.Head(
          address=row[6],
          arc_score=0.0),
        label=row[7]
        
      )
      if not row[5] == "_":
        feats = row[5].split("|")
        for feat in feats:
          morph_feat = token.morphology.add(
            name=feat.split("=")[0].lower(),
            value=feat.split("=")[1].lower()
            )
    sentence.text = " ".join([token.word for token in sentence.token if token.word != "ROOT"])
    print(sentence.text)
    # finally we add semantic roles
    if not semantic_roles:
      return sentence
    else:
      predicates = {}
      predicate_counter = 0
    # input("...")
    for index, row in df.iterrows():
      if row[10] == "Y":
        predicate_counter += 1
        arg_str = sentence.argument_structure.add(
          predicate=row[11],
          predicate_index=row[0]
        )
        srl_roles_column = 11+predicate_counter
        for val in df[srl_roles_column]:
          if val != "_":
            srl = val
            argument = arg_str.argument.add(srl=srl)
            idx = df[srl_roles_column][df[srl_roles_column] == srl].index
            for id_ in idx:
              token_index = df.loc[id_, 0]
              span_indices = nn_utils.get_argument_spans(
                sentence,
                token_index,
                arg_str.predicate_index,
                [])
              argument_span = sorted(set([token_index] + span_indices))
              argument.token_index.extend(argument_span)

    # print(text_format.MessageToString(sentence, as_utf8=True))
    # input("...")
    return sentence

def main(args):
  if args.propbank:
    converter = PropbankConverter(args.input_file)
    sentence_protos = []
    conll_df_list = converter._ReadCoNLLDataFrame(converter._corpus)
    for df in conll_df_list:
      df = df.dropna(how='all').reset_index(drop=True)
      cols = [0, 6]
      df = df.astype({c: int for c in cols})
      if not df.empty:
        sentence_proto = converter._DataFrameToProto(df)
        sentence_protos.append(sentence_proto)
  else:
    # if propbank is false, we only convert dependency treebank without any 
    # semantic roles.
    converter = Converter(args.input_file)
    # input("Press to continue..")
    sentences = converter.sentence_list
    # random_sentences = random.sample(sentences, 50)
    sentence_protos = converter.ConvertConllToProto(conll_sentences=sentences)

  converter.Write(sentence_protos, output_file=args.output_file,
                  writetext=args.writetext, writeproto=args.writeproto,
                  prototype=args.prototype)
    #written_proto = reader.ReadSentenceProto(args.output_file)
    # DO NOT DELETE THIS FOR THE SAKE OF FUTURE REFERENCE
    #print(text_format.MessageToString(written_proto, as_utf8=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        help="Path to input file.")
    parser.add_argument("--output_file",
                        help="The output file to write the data.")
    parser.add_argument("--writetext",
                        help="wheter to save the output also in .pbtxt format", default=True)
    parser.add_argument("--writeproto",
                        help="whether to save the output in proto format.", default=False)
    parser.add_argument("--prototype",
                        help="whether sentence or treebank proto.", default="treebank")
    parser.add_argument("--propbank", type=bool,
                        help="whether to convert propbank.", default=False)
    args = parser.parse_args()
    main(args)
