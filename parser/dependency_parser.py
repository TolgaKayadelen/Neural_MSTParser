# -*- coding: utf-8 -*-

"""Dependency Parser."""


import numpy as np
from learner.perceptron import ArcPerceptron
from data.treebank import sentence_pb2
from decoder import Decoder
from google.protobuf import text_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from mst.max_span_tree import GetTokenByAddressAlt
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

class DependencyParser:
    def __init__(self, feature_opts={}, decoding="mst"):
        self.feature_opts = feature_opts
        self.arc_perceptron = ArcPerceptron(self.feature_opts)
        self.decoder = Decoder(decoding)
        self.feature_extractor = FeatureExtractor()
        self.arc_accuracy = None
    
    def MakeFeatures(self, training_data):
        """Makes all features from the training data.
        
        Args:
            training_data: list, list of sentence_pb2.Sentence() objects.
        """
        self.arc_perceptron.MakeAllFeatures(training_data)
    
    def Parse(self, sentence):
        """Parse a sentence.
        
        Args:
            sentence: sentence_pb2.Sentence(), without head-dependent relations.
        
        Returns:
            parsed: sentence_pb2.Sentence(), parsed sentence.
            predicted_heads: list, list of predicted heads. 
        """
        assert sentence.token[0].index == -1 and sentence.token[-1].index == -2
        assert sentence.HasField("length"), "Sentence must have a length."
        score_matrix = np.zeros((sentence.length, sentence.length))
        #print(score_matrix.shape)
        #w = 0
        for token in sentence.token:
            for ch in token.candidate_head:
                head = GetTokenByAddressAlt(sentence.token, ch.address)
                #print("child {}".format(token.word))
                #print("head {}".format(head.word))
                features = self.feature_extractor.GetFeatures(
                    sentence = sentence,
                    head = head,
                    child = token,
                    use_tree_features=True
                )
                #w+=1
                score = self.arc_perceptron.Score(features)
                score_matrix[token.index][head.index] = score
                #print("score {}".format(score))
                #print("--------------------------------------")
        #print(score_matrix)
        probs = self._Softmax(score_matrix)
        #print(probs)
        #print(probs.sum(axis=1))
        parsed, predicted_heads = self.decoder(sentence, probs)
        return predicted_heads
        
    def Train(self, niters, training_data, dev_data=None, approx=1):
        """Train the arc perceptron."""
        for i in range(niters+1):
            #Train arc perceptron for one epoch.
            nr_correct_heads, nr_childs = self.arc_perceptron.Train(training_data)
            #Evaluate the arc perceptron
            #accuracy = nr_correct_heads * 100 / nr_childs
            #print("accuracy after iter {} = %{}".format(i, accuracy))
            #print(training_data[:approx])
            train_acc = self.Evaluate(training_data[:approx])
            logging.info("Train acc after iter {}: {}".format(i, train_acc))
            #dev_acc = self.Evaluate(dev_data[:approx])
            if train_acc == 100:
                break
            np.random.shuffle(training_data)
    
    def Evaluate(self, eval_data):
        """Evaluates the performance of arc perceptron on data.
        
        Args:
            eval_data = list, list of sentence_pb2.Sentence() objects.
        Returns:
            accuracy of the perceptron on the dataset.
        
        """
        acc = 0.0
        print(len(eval_data))
        for sentence in eval_data:
            assert sentence.token[0].index == -1 and sentence.token[-1].index == -2
            assert sentence.HasField("length"), "Sentence must have a length!" 
            predicted_heads = self.Parse(sentence)
            # get the gold heads for tokens except for the dummy ones.
            gold_heads = [token.selected_head.address for token in sentence.token[1:-1]]
            #print(predicted_heads, gold_heads)
            assert len(predicted_heads) == len(gold_heads), "Number of predicted and gold heads don't match!!!"
            acc += self._Accuracy(predicted_heads, gold_heads)
        return acc / len(eval_data)
    
    
    def _Accuracy(self, predicted, gold):
        return 100 * sum(ph == gh for ph, gh in zip(predicted, gold)) / len(predicted)
        

    def _Softmax(self, matrix):
      """Numpy softmax function (normalizes rows)."""
      matrix -= np.max(matrix, axis=1, keepdims=True)
      #print(matrix)
      matrix = np.exp(matrix)
      #print(matrix)
      return matrix / np.sum(matrix, axis=1, keepdims=True)    


def main():
    parser = DependencyParser()
    #extractor = FeatureExtractor(filename="./learner/features.txt")
    test_sentence = reader.ReadSentenceTextProto("./data/testdata/parser/john_saw_mary.pbtxt")
    test_sentence = common.ConnectSentenceNodes(test_sentence)
    parser.MakeFeatures([test_sentence])
    init_w = 0.1
    for key in parser.arc_perceptron.weights.keys():
        for value in parser.arc_perceptron.weights[key].keys():
            parser.arc_perceptron.weights[key][value] += init_w
            init_w += 0.1
    parser.Train(3, [test_sentence])
    #parser.Parse(test_sentence)
    #common.PPrintWeights(parser.arc_perceptron.weights)


if __name__ == "__main__":
    main()