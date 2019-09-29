# -*- coding: utf-8 -*-

"""Dependency Labeler."""

from learner.perceptron import LabelPerceptron
from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


# A dependency labeler works on the output of the dependency parser to label
# the dependency arcs assingned by the parser to the heads and dependents.
class DependencyLabeler:
    def __init__(self, feature_opts={}):
        self.feature_opts = feature_opts
        self.feature_extractor = FeatureExtractor("labelfeatures")
        self.label_accuracy = None

    def MakeFeatures(self, training_data):
        """Makes features from the training data.

        Note that for dependency labeler, the relevant features are features
        from the gold head-dependent pairs.
        Args:
            training_data: list, list of sentence_pb2.Sentence() objects.
        """
        self.label_perceptron.MakeAllFeatures(training_data)

    def Label(self, sentence):
        """Label the dependency arcs in a sentence.
        Args:
            sentence: sentence_pb2.Sentence(), without dependency labels.
        Returns:
            sentence: sentence_pb2.Sentence(), with dependency labels.
        """
        assert sentence.HasField("length"), "Sentence must have length!!"
        for token in sentence.token:
            head = common.GetTokenByAddress(sentence.token, token.selected_head.address)
            features = self.feature_extractor.GetFeatures(
                sentence=sentence,
                head=head,
                child=token
            )
            token.label = self.label_perceptron._PredictLabel(token, features)
        return sentence

    def Train(self, training_data, test_data=None, approx=10):
        """Train the label perceptron."""
        for i in range(niters):
            print("\n**************-------------------*************")
            logging.info("Starting LP Training Epoch {}".format(i+1))
            #Train label perceptron for one epoch.
            self.label_perceptron.Train(training_data)
            #Evaluate the label perceptron
            train_acc = self._Evaluate(training_data)
            logging.info("LP train acc after iter {}: {}".format(i+1, train_acc))
            raw_input("Press a key to continue: ")
            if test_data:
                logging.info("Evaluating LP on test data..")
                test_acc = self._Evaluate(test_data)
                # Comment out if you're not interested in seeing test acc after
                # each epoch.
                logging.info("LP Test acc after iter {}: {}".format(i+1, test_acc))
                raw_input("Press a key to continue: ")
            #if train_acc == 100:
            #    break
            np.random.shuffle(training_data)

    def _Evaluate(self, eval_data):
        """Evaluates the performance of label perceptron on data.
        Args:
            eval_data = list, list of sentence_pb2.Sentence() objects.
        Returns:
            accuracy of the label perceptron on the dataset.
        """
        acc = 0.0
        #print(len(eval_data))
        for sentence in eval_data:
            gold_labels = [token.label for token in sentence.token]
            assert sentence.HasField("length"), "Sentence must have a length!"
            #common.PPrintTextProto(sentence)
            # clear the label field
            for token in sentence.token:
                token.ClearField("label")
            # label the sentence with the model
            labeled = self.Label(sentence)
            predicted_labels = [token.label for token in sentence.token]
            logging.info("predicted {}, gold {}".format(predicted_labels, gold_labels))
            assert len(predicted_labels) == len(gold_labels), """Number of
                predicted and gold labels don't match!!!"""
            acc += self._Accuracy(predicted_labels, gold_labels)
        return acc / len(eval_data)

    def _Accuracy(self, prediction, gold):
        return 100 * sum(pl == gl for pl, gl in zip(prediction, gold)) / len(prediction)

def main():
    labeler = DependencyLabeler()

if __name__ == "__main__":
    main()
