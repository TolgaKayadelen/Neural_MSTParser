# -*- coding: utf-8 -*-

"""Common utility functions."""
#TODO: unit tests for all functions here. 

from util import reader
from data.treebank import sentence_pb2
from copy import deepcopy

#type: sentence util
def ExtendSentence(sentence):
    """Adds a dummy start and end node (tokens) to the sentence.
    
    Args: 
        sentence: sentence_pb2.Sentence()
    Returns:
        sentence: sentence_pb2.Sentence()
    """
    start_token = sentence_pb2.Token(word="START_TOK", lemma="START_TOK", category="START_POS", pos="START_POS", index=-1)
    end_token = sentence_pb2.Token(word="END_TOK", lemma="END_TOK", category="START_POS", pos="START_POS", index=-2)
    tokens = deepcopy(sentence.token)
    del sentence.token[:]
    sentence.token.extend([start_token])
    for token in tokens:
        sentence.token.extend([token])
    sentence.token.extend([end_token])
    return sentence

# type: sentence util
def GetRightMostChild(sentence, token):
    """Returns the rightmost child of token in sentence. 
    
    Args: 
        token, token.pb2 object. The token whose child we want. 
        sentence, sentence.pb2 object. The sentence we search to get the child of token. 
    Returns:    
        rightmost = the rightmost child token.
    """
    assert token.HasField("index"), "Token doesn't have index."
    children = [] 
    max_ = 0
    rightmost = None
    for child in sentence.token:
        #print("evalutating child {}".format(child.word))
        #print("selected head for this child {}".format(child.selected_head.address))
        if child.selected_head.address == token.index:
            if child.index > max_:
                rightmost = child
            max_ = child.index
    #return max(children, key=lambda x:x.index)
    return rightmost


#type: sentence util
def GetBetweenTokens(sentence, head, child, dummy):
    """Returns a list of tokens between the head and the child"""
    assert head.HasField("index") and child.HasField("index"), "Token has no index"
    if head.index > child.index:
        btw_tokens = sentence.token[child.index+1:head.index]
    else:
        btw_tokens = sentence.token[head.index+1+dummy:child.index+dummy]
    return btw_tokens if btw_tokens else [None]


def GetValue(token, feat):
    """Returns the desired value for a feature of a token.
    
    Args:
        token: sentence_pb2.Token()
        feat: string, the feature for which we want the value. 
       
    Returns: the desired value for a feature of the token
    """
    if token == None:
        return "None"
    elif token.index == 0 or token.pos == "TOP":
        return "ROOT"
    else:
        if feat == "word":
            return token.word.encode("utf-8")
        if feat == "pos":
            return token.pos.encode("utf-8")
        if feat == "lemma":
            return token.lemma.encode("utf-8")

if __name__ == "__main__":
    # TODO: write proper tests for this. 
    sentence = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
    token = sentence.token[2] # saw
    assert token.word == "saw"
    rightmost = GetRightMostChild(sentence, token)
    print("rightmost child is {}".format(rightmost.word))
                