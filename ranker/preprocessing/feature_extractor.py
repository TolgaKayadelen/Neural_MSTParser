# -*- coding: utf-8 -*-

"""Feature Extractor for the Dependency Parser and Dependency Labeler.

Feature representation is inspired from:
McDonald, R., Crammer, K., Pereira, F. (2006). Online Large-Margin Training of Dependency Parsers.

"""
from ranker.preprocessing import feature_pb2
from data.treebank import sentence_pb2

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

FEATURE_DIR = "ranker/features"
TEST_FEATURE_DIR = "data/testdata/features/test_features"

class FeatureExtractor:

    def _get_surrounding_token_features(self, current_index, distance, sentence):
      feature = feature_pb2.Feature()
      token = sentence[current_index+distance]
      if distance < 0 and token.index > current_index:
        raise IndexError(f"Index {current_index+distance} out of reach with current token index {current_index}")
      feature.word = token.word
      feature.pos = token.pos
      feature.category = token.category
      feature.lemma = token.lemma
      feature.label = token.label
      for morph in token.morphology:
        feature.morphology.add(name=morph.name, value=morph.value)
      return feature

    def get_features(self, token, sentence, n_prev, n_next):
        """Extracts features for a token.
        Args:
            sentence: the list of tokens in the sentence.
            token: sentence_pb2.Token, the head token.
            n_prev: int, number of previous tokens to extract features from
            n_next: int, number of next tokens to extract features from
        Returns:
            featureset: featureset_pb2.FeatureSet(), a proto of feature names and values.
            Note that this doesn't return any weight for the features.
        """
        if n_prev > 0:
          raise ValueError("n_prev cannot be positive!")
        feature = feature_pb2.Feature()
        feature.word = token.word
        feature.pos = token.pos
        feature.category = token.category
        feature.lemma = token.lemma
        feature.label = token.label
        for morph in token.morphology:
          feature.morphology.add(name=morph.name, value=morph.value)
        if n_prev:
          for i in range(-1, n_prev-1, -1):
            try:
              prev = self._get_surrounding_token_features(token.index, distance=i, sentence=sentence)
            except IndexError:
              break
            previous_token = feature.previous_token.add()
            previous_token.CopyFrom(prev)
            previous_token.distance = i
        if n_next:
          for i in range(1, n_next+1, 1):
            try:
              next = self._get_surrounding_token_features(token.index, distance=i, sentence=sentence)
            except IndexError:
              break
            next_token = feature.next_token.add()
            next_token.CopyFrom(next)
            next_token.distance = i
        # print(feature)
        return feature
