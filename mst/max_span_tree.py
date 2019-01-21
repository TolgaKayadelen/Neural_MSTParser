# -*- coding: utf-8 -*-

"""Functions for returning the maximum spanning tree of a sentence. 

The algorithm used for finding the MST is Chu-Liu-Edmonds.
"""

import argparse
from collections import defaultdict
from data.treebank import sentence_pb2 
from util import reader

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

    
def ChuLiuEdmonds(sentence):
    """Implements the Chi-Liu-Edmonds algorithm to find the MST. 
    
    Args:
        sentence_proto_path: path to a protocol buffer Sentence object.
    Returns:
        The maximum spanning tree of the sentence.
    """
    if not sentence.HasField("length"):
        sentence.length = len(sentence.token)
        #print(sentence.length)
    mst = sentence_pb2.maximum_spanning_tree()
    mst.sentence.CopyFrom(_GreedyMst(sentence))
    #print(mst.sentence)
    
    # Find if there are any cycles in the returned MST, if so contract.
    cycle, cycle_path = _Cycle(mst.sentence)
    if not cycle:
        _DropCandidateHeads(mst.sentence)
        return mst.sentence
    
    new_token, original_edges, contracted = _Contract(mst.sentence, cycle_path)
    reconstructed = _Reconstruct(ChuLiuEdmonds(contracted), new_token, original_edges, cycle_path)
    return reconstructed


def _GreedyMst(sentence):
    """Find the MST of the sentence using greedy search.
    
    The MST of the sentence is the sum of best incoming arcs for each token. So 
    for each candidate head in each token we find the one that has the highest
    score and write it to the selected head. 
    This function does not guarantee that the resulting MST will be cycle free. If 
    there are cycle, they are handled later by the Contact function.
    args:
        sentence, a protocol buffer object representation of a sentence.
    returns:
        sentence where the selected heads are computed based on argmax of 
        candidate heads. 
    """
    
    for token in sentence.token:
        if token.index == 0:
            continue
        max_arc_score = 0.0
        max_head = 0
        for ch in token.candidate_head:
            if ch.arc_score > max_arc_score:
                max_head = ch.address
                max_arc_score = ch.arc_score
        token.selected_head.address = max_head
        token.selected_head.arc_score = max_arc_score
    return sentence
                

def ConnectSentenceNodes(sentence):
    """Connect the tokens (nodes) of a sentence to every other token.
    
    Every token has every other token as a head candidate, except for itself. 
    The ROOT token has no head candidates. 
    
    Args:
        sentence: A protocol buffer Sentence object. 
    Returns:
        The sentence where all tokens are connected.
    """
    token_connections = [
        (i, j) 
        for i in sentence.token for j in sentence.token[::-1] 
        if i.word != j.word and i.word.lower() != "root"
        ]
    #for edge in edges:
    #    print("child ({0} {1}), head: ({2} {3})".format(
    #        GetTokenIndex(sentence.token, edge[0]),
    #        #type(edge[0].word),
    #        edge[0].word.encode("utf-8"),
    #        GetTokenIndex(sentence.token, edge[1]),
    #        #type(edge[1].word)
    #        edge[1].word.encode("utf-8")
    #    ))
    
    for edge in token_connections:
        # ch: candidate_head
        ch = edge[0].candidate_head.add()
        #ch.address = _GetTokenIndex(sentence.token, edge[1])
        ch.address = edge[1].index
        ch.arc_score = 0.0 # default
    
    # Sanity checking:
    # Each token should have sentence.length - 1 number of candidate heads, i.e. 
    # all other tokens except for itself. The root token (indexed 0) should not
    # have any candidate heads. 
    for token in sentence.token:
        if _GetTokenIndex(sentence.token, token) == 0:
            assert len(token.candidate_head) == 0
        else:
            assert len(token.candidate_head) == len(sentence.token) - 1
    return sentence
    

def _DropCandidateHeads(sentence):
    """Removes the candidate head fields from a sentence where mst is computed. 
    Args:
        sentence, sentence in mst.sentence protocol buffer representation. 
    """
    for token in sentence.token:
        token.ClearField("candidate_head")


def _ClearSelectedHeads(sentence):
    """Removes the selected head fields from a sentence that is contracted.
    
    This is useful just for clarift. After contraction, selected heads will be
    re-determined with ChuLiuEdmonds based on recalculated arc scores. 
    
    Args:
        sentence: sentence_pb2.Sentence() from which we remove selected_heads in tokens.
    
    Returns:
        sentence where tokens are cleared of selected_heads.
    """
    for token in sentence.token:
        if not token.index == 0:
            token.ClearField("selected_head")
    return sentence

def _Cycle(sentence):
    """Checks whether there is a cycle in the sentence.
    
    args:
        sentence: a protocol buffer Sentence object.

    returns:
        cycle: boolean, whether the sentence has cycle.
        path: list, start and end indexes of the cycle.
    """
    
    p = []
    # start iterating from token 1 since 0 is ROOT with no head.
    for token in sentence.token[1:]:
        #print(_GetTokenIndex(sentence.token, token))
        cycle, cycle_path = _GetCyclePath(token, sentence, p)
        #print(cycle)
        del p[:]
        if cycle == True:
            break
    return cycle, cycle_path
    
def _GetCyclePath(start_token, sentence, p):
    """Returns the cycle path for a sentence which has a cycle.
    
    This function traverses the tree from the start_token parameter to check
    if there's a cycle in the tree.
    
    Args:
        start_token: token, the token from which the tree traversal starts.
        sentence: a protocol buffer Sentence to traverse.
        p: list
    Returns:
        cycle: boolean, whether the tree contains a cycle.
        path: list, the path of the cycle that exists.
    """
    path = p
    tokens = sentence.token
    token = start_token
    path.append(_GetTokenIndex(tokens, token))
    #print(path)
    # base case: if the token is ROOT, there can't be no more haed to visit.
    if token.selected_head.address == -1:
        return False, path
    # base case: if the token is already in the visited list of heads, there's cycle.
    if token.selected_head.address in path:
        path.append(token.selected_head.address)
        cycle_start_index = path.index(token.selected_head.address)
        cycle_path = path[cycle_start_index:]
        # print("There is cycle: ", cycle_path)
        return True, cycle_path
    # recursive step
    #print("word: ", token.word, "address: ", token.selected_head.address)
    next_token = _GetTokenByAddressAlt(tokens, token.selected_head.address)
    #print("next token: ", next_token)
    return _GetCyclePath(next_token, sentence, p=path)
    

def _Contract(sentence, cycle_path):
    """Contracts the cycle in the sentence. 
    
    Args: 
        sentence: a protocol buffer Sentence object.
        cycle_path: list, the path of the cycle. 
    """
    if not sentence.length:
        sentence.length = len(sentence.token)
    cycle_tokens, cycle_score = _GetCycleScore(sentence, cycle_path)
    outcycle_tokens = [token for token in sentence.token if not token in cycle_tokens]
    new_token_index = sentence.token[-1].index + 1
    new_token = sentence.token.add()
    new_token.index = new_token_index
    new_token.word = "cycle_token_" + str(new_token_index)
    sentence.length += 1
    original_edges = _GetOriginalEdges(sentence)
    _RedirectIncomingArcs(cycle_tokens, new_token, cycle_score)
    _RedirectOutgoingArcs(cycle_tokens, outcycle_tokens, new_token)
    pruned = _DropCycleTokens(sentence, cycle_tokens)
    contracted = _ClearSelectedHeads(pruned)
    contracted.length = len(contracted.token)
    #print(contracted)
    return new_token, original_edges, contracted
    
       
def _GetCycleScore(sentence, cycle_path):
    """Returns the score and the tokens of the cycle in the sentence.
    
    Args:
        sentence: a protocol buffer Sentence object.
        cycle_path: list, path of the cycle. 
    Returns:
        cycle_tokens: list, list of sentence_pb2.Token() that constitute that cycle.
        cycle_score: float, the weight of the cycle in Sentence.
    """
    cycle_tokens = [sentence.token[i] for i in set(cycle_path)]
    #print(cycle_tokens)
    cycle_score = 0.0
    for ind in set(cycle_path):
        token = sentence.token[ind]
        if token.selected_head.address == ind:
            break
        cycle_score += token.selected_head.arc_score
    #print("cycle_score: ", cycle_score)
    return cycle_tokens, cycle_score

def _GetOriginalEdges(sentence):
    """Creates some convenience information about the edges and edge scores. 
    
    This information will be useful when reconstructing the graph after
    _Contract if a cycle is found. 
    
    Args: 
        sentence: sentence_pb2.Sentence(), the sentence to get edge information for.
    Returns:
        original_edges: defaultdict, dict of edges and edge scores.
    """
    original_edges = defaultdict(list)
    for token in sentence.token:
        if token.index == 0 or token.selected_head.address == -1:
            continue
        for ch in token.candidate_head:
            original_edges[ch.address].append((token.index, ch.arc_score))
    #print(original_edges)
    return original_edges
    
def _RedirectIncomingArcs(cycle_tokens, new_token, cycle_score):
    """Redirect any incoming arcs to the cycle to the new target token.
    
    Add each token address that is the head of a token inside the cycle
    as candidate head to the new token. This way we make sure that each
    token outside the cycle that has a dependant in the cycle is now the
    head of this new token. Also recalculate the arc scores.
    """
    for token in cycle_tokens:
        for candidate_head in token.candidate_head:
            if candidate_head.address != token.selected_head.address:
                ch = new_token.candidate_head.add()
                ch.address = candidate_head.address
                ch.arc_score = candidate_head.arc_score - token.selected_head.arc_score + cycle_score
    #print(new_token)        

def _RedirectOutgoingArcs(cycle_tokens, outcycle_tokens, new_token):
    """Redirect any outgoing arcs from the cycle to the new target token. 
    
    For each token outside the cycle that has as head one of the tokens
    inside the cycle, craete an arc between the new_target to the token
    where the new target is the head of the token outside the cycle and
    set the arc score appropriately. 
    
    
    Args:   
        cycle_tokens: the tokens inside the cycle.
        outcycle_tokens: the tokens outside the cycle. 
        new_token: the new target token. 
    """
    cycle_token_indexes = [token.index for token in cycle_tokens]
    for token in outcycle_tokens:
        if not token.candidate_head:
            continue
        if token == new_token:
            continue        
        for candidate_head in token.candidate_head:
            if candidate_head.address in cycle_token_indexes:
                candidate_head.address = new_token.index
                    

def _DropCycleTokens(sentence, cycle_tokens):
    """Drops cycle tokens from the sentence."""
    
    for token in cycle_tokens:
        sentence.token.remove(token)
    return sentence

def _Reconstruct(sentence, new_token, original_edges, cycle_path):
    """Reconstruct.
    
    Args:
        sentence: A contracted sentence from which to reconstruct MST.
    
    """
    #print("Printing in reconstruct function: ")
    print(sentence)
    candidate_sources, candidate_scores = [], []
    for token in sentence.token:
        # Handle outgoing edges from the cycle to outside the cycle.
        if token.selected_head.address == new_token.index:
            for source, targets in original_edges.items():
                for target in targets:
                    if target[0] == token.index:
                        candidate_sources.append(source)
                        candidate_scores.append(target[1])
            print(candidate_sources, candidate_scores)
        #del candidate_sources[:]
        #del candidate_scores[:]        


def _SentenceWeight(sentence):
    """Computes the total weight of the sentence."""
    pass

def _GetTokenIndex(tokens, token):
    """Return the index of a token in the sentence.
    
    Note that this is safer than calling token.index from the proto and
    should be preferred where possible. The reason is that index is an
    optional field and might not have been initialized at all. In that
    case the proto returns 0, which can break the code. 
    
    Args:
        tokens: list of tokens in a protocol buffer Sentence object.
        token: the token of which we want to get the index.
    Returns:
        int, the index of the token.
    """
    return list(tokens).index(token)


def _GetTokenByAddress(tokens, address):
    """Returns the token in this address.
    args: 
        tokens: list of tokens
        address: the address of the token which we want.
    returns:
        the token in the given address.
    """
    return tokens[address]


def _GetTokenByAddressAlt(tokens, address):
    """Alternative to the above function."""
    #TODO: Once it is clear that everything works fine, kill the other function
    # and change the code as necessary.
    list_indices = [] 
    for token in tokens:
        assert token.HasField("index"), "Token doesn't have index."
        list_indices.append(token.index)
        assert list_indices.count(token.index) == 1, "Can't have two tokens with same index."
        if token.index == address:
            found = token
    return found


def main(args):
    sentence = reader.ReadSentenceProto(args.input_file)
    ConnectSentenceNodes(sentence)
    ChuLiuEdmonds(sentence)
    #sentence = sentence_pb2.Sentence()
    #with open(args.input_file, "rb") as sentence_proto:
    #    sentence.ParseFromString(sentence_proto.read())
    #    print(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to input file.")
    args = parser.parse_args()
    main(args)
