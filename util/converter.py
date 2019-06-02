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
bazel-bin/data/treebank/converter 
--input_file=/path/to/input/conll/corpus 
--output_file=/path/to/output_file # do not give file extension
--writetext=True
--writeproto=True

Example:
bazel-bin/data/treebank/converter 
--input_file=./data/UDv23/UD_Turkish_IMST/tr_imst_ud_dev.conllu 
--output_file=./data/treebank/sentence_5 
--writetext=True


"""

from __future__ import print_function
from data.treebank import sentence_pb2
from util import reader, writer
from collections import defaultdict, OrderedDict
from copy import deepcopy
from google.protobuf import text_format

import argparse
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


class Converter:
    
    def __init__(self, corpus):
        self._corpus = corpus
        #self.sentence_list = self._ReadSentences(self._corpus)
        self.sentence_list = reader.ReadConllX(self._corpus)
        
    def ConvertConllToProto(self, conll_sentences, output_file, writetext, writeproto, keep_igs=True):
        """Converts conll-X formatted sentences to proto buffer objects.    
        Args:
            dep_graphs: list of dependency graphs.
        Returns:    
            list of proto buffers.
        """
        sentence_protos = []
        logging.debug("Converting sentences to protocol buffer.")
        for conll in conll_sentences:
            sentence = sentence_pb2.Sentence()
            sentence_dict, metadata = self._ConvertToDict(conll)
            
            # add metadata
            sentence.text = metadata["text"]
            sentence.sent_id = metadata["sent_id"]
            
            # add a root token
            root_token = sentence.token.add()
            root_token.word = "ROOT"
            root_token.category = "TOP"
            root_token.pos = "TOP"
            root_token.selected_head.address = -1
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
            
            # finally add sentence length
            sentence.length = len(sentence.token)
            sentence_protos.append(sentence)
    
        assert len(conll_sentences) == len(sentence_protos)
        logging.debug("%d sentences converted to protocol buffer." % len(conll_sentences))
        
        if writetext:
            text_output = output_file + ".pbtxt"
            for sentence_proto in sentence_protos:
                writer.write_proto_as_text(sentence_proto, text_output)
        if writeproto:
            proto_output = output_file + ".protobuf"
            for sentence_proto in sentence_protos:
                writer.write_proto_as_proto(sentence_proto, proto_output)    
        
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
        Returns:
            sentence mapped to a defaultdict.
            metadata: the text and the id of the sentence.
        """
        sentence_dict = defaultdict(OrderedDict)
        metadata = {}
        token = 0
        for line in sentence:
            if line.startswith("# sent_id"):
                metadata["sent_id"] = line.split("=")[1].strip()
                continue
            if line.startswith("# text"):
                metadata["text"] = line.split("=")[1].strip()
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
        

def main(args):
    converter = Converter(args.input_file)
    sentences = [converter.sentence_list[4]]
    protos = converter.ConvertConllToProto(
        conll_sentences = sentences, 
        output_file = args.output_file, 
        writetext = args.writetext, 
        writeproto = args.writeproto
        )
    #written_proto = reader.ReadSentenceProto(args.output_file)
    # DO NOT DELETE THIS FOR THE SAKE OF FUTURE REFERENCE
    #print(text_format.MessageToString(written_proto, as_utf8=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to input file.")
    parser.add_argument("--output_file", help="The output file to write the data.")
    parser.add_argument("--writetext", help="wheter to save the output also in .pbtxt format", default=True)
    parser.add_argument("--writeproto", help="whether to save the output in proto format.", default=False)
    args = parser.parse_args()
    main(args)
