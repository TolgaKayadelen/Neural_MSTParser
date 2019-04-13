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
        mst, sentence_pb2.maximum_spanning_tree. Gives the mst and the mst score
            of the sentence.
    """
    logging.info("Processing sentence --> {}\n".format(" ".join([token.word for token in sentence.token[1:]])))
    if not sentence.HasField("length"):
        sentence.length = len(sentence.token)
        #logging.info("Sentence length: {}".format(sentence.length))
    mst = sentence_pb2.maximum_spanning_tree()
    mst.sentence.CopyFrom(_GreedyMst(sentence))
    
    # Find if there are any cycles in the returned MST, if so contract.
    cycle, cycle_path = _Cycle(mst.sentence)
    if not cycle:
        mst.sentence.CopyFrom(_DropCandidateHeads(mst.sentence))
        mst.score = _GetSentenceWeight(mst.sentence)
        return mst
    
    new_token, original_edges, contracted = _Contract(mst.sentence, cycle_path)
    #print("cycle_path {}".format(cycle_path))
    reconstructed_edges = _Reconstruct(
        ChuLiuEdmonds(contracted),
        new_token,
        original_edges,
        cycle_path
        )
    reconstructed_sentence = _Merge(reconstructed_edges, sentence)
    #logging.info("Reconstructed Edges: {}".format(reconstructed_edges))
    mst.sentence.CopyFrom(reconstructed_sentence)
    assert mst.sentence.length == len(mst.sentence.token)
    mst.score = _GetSentenceWeight(mst.sentence)
    return mst


def _GreedyMst(sentence):
    """Find the MST of the sentence using greedy search.
    
    The MST of the sentence is the sum of best incoming arcs for each token. So 
    for each candidate head in each token we find the one that has the highest
    score and write it to the selected head. 
    This function does not guarantee that the resulting MST will be cycle free. If 
    there are cycles, they are handled later by the _Contact() function.
    
    args:
        sentence, a sentence_pb2.Sentence object.
    returns:
        sentence where the selected heads are computed based on argmax of 
        candidate heads. 
    """
    
    for token in sentence.token:
        if token.index == 0:    # the 0th token is the ROOT, has no head.
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
    
    for edge in token_connections:
        # ch: candidate_head
        ch = edge[0].candidate_head.add()
        # ch.address = _GetTokenIndex(sentence.token, edge[1])
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
    Returns: 
        sentence where the candidate_head field is cleared from tokens.
    """
    for token in sentence.token:
        token.ClearField("candidate_head")
    return sentence

def _ClearSelectedHeads(sentence):
    """Removes the selected head fields from a sentence that is contracted.
    
    This is useful just for clarity. After contraction, selected heads will be
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
        sentence: a sentence_pb2.Sentence object.

    returns:
        cycle: boolean, whether the sentence has cycle.
        path: list, start and end indexes of the cycle.
    """
    logging.info("Checking for cycles...")
    p = []
    # start iterating from token 1 since 0 is ROOT with no head.
    for token in sentence.token[1:]:
        #print("token is {}".format(token.word))
        cycle, cycle_path = _GetCyclePath(token, sentence, p)
        #print("cycle path: ", cycle, cycle_path)
        del p[:]
        if cycle == True:
            #print("breaking... ")
            break
    return cycle, cycle_path
    
def _GetCyclePath(start_token, sentence, p):
    """Returns the cycle path for a sentence which has a cycle.
    
    This function traverses the tree from the start_token parameter to check
    if there's a cycle in the tree.
    
    Args:
        start_token: token, the token from which the tree traversal starts.
        sentence: a sentence_pb2.Sentence object to traverse.
        p: list
    Returns:
        cycle: boolean, whether the tree contains a cycle.
        path: list, the path of the cycle that exists.
    """
    path = p
    #print("path: {}".format(path))
    tokens = sentence.token
    token = start_token
    path.append(token.index)
    logging.info("path: {}".format(path))
    # base case: if the token is ROOT, there can't be no more head to visit.
    if token.selected_head.address == -1:
        return False, path
    # base case: if the token is already in the visited list of heads, there's cycle.
    if token.selected_head.address in path:
        path.append(token.selected_head.address)
        #(TODO): the following only works because of the way the python interpreter works.
        # When there are multiple occurences of the same item in a list, list.index(item)
        # returns the index for the first occurence. Consider moving to a safer 
        # implementation.
        cycle_start_index = path.index(token.selected_head.address)
        cycle_path = path[cycle_start_index:]
        logging.info("There is cycle in the sentence, cycle_path: {}".format(cycle_path))
        return True, cycle_path
    # recursive step
    #logging.info("""No cycle found yet until word: {}, checking for next head: {} """.format(
    #    token.word, token.selected_head.address))
    next_token = _GetTokenByAddressAlt(tokens, token.selected_head.address)
    #logging.info("Next token for recursion: {}".format(next_token))
    return _GetCyclePath(next_token, sentence, p=path)
    

def _Contract(sentence, cycle_path):
    """Contracts the cycle in the sentence. 
    
    Args: 
        sentence: a sentence_pb2.Sentence object.
        cycle_path: list, the path of the cycle. 
    """
    #logging.info("Contracting the cyclic sentence...")
    if not sentence.length:
        sentence.length = len(sentence.token)
    # Get the score of the cycle and the tokens that make up the cycle.
    cycle_tokens, cycle_score = _GetCycleScore(sentence, cycle_path)
    # Get the tokens that are outside the cycle.
    outcycle_tokens = [token for token in sentence.token if not token in cycle_tokens]
    # Add a new token to the tree to represent the cycle.
    new_token_index = sentence.token[-1].index + 1
    new_token = sentence.token.add()
    new_token.index = new_token_index
    new_token.word = "cycle_token_" + str(new_token_index)
    sentence.length += 1
    # Register the original edges in the sentence. We will need this when we want to
    # reconstruct the sentence after breaking the cycle.
    original_edges = _GetOriginalEdges(sentence)
    # Redirect the incoming and outgoing arcs to the new token. 
    _RedirectIncomingArcs(cycle_tokens, new_token, cycle_score)
    _RedirectOutgoingArcs(cycle_tokens, outcycle_tokens, new_token)
    pruned = _DropCycleTokens(sentence, cycle_tokens)
    contracted = _ClearSelectedHeads(pruned)
    contracted.length = len(contracted.token)
    return new_token, original_edges, contracted
    
       
def _GetCycleScore(sentence, cycle_path):
    """Returns the score and the tokens of the cycle in the sentence.
    
    Args:
        sentence: a protocol buffer Sentence object.
        cycle_path: list, path of the cycle. 
    Returns:
        cycle_tokens: list, list of sentence_pb2.Token() objects that constitute a cycle.
        cycle_score: float, the weight of the cycle.
    """
    cycle_tokens = [token for token in sentence.token if token.index in set(cycle_path)]
    #print(cycle_tokens)
    cycle_score = 0.0
    for ind in set(cycle_path):
        #token = sentence.token[ind]
        token = _GetTokenByAddressAlt(sentence.token, ind)
        if token.selected_head.address == ind:
            break
        cycle_score += token.selected_head.arc_score
    return cycle_tokens, cycle_score

def _GetOriginalEdges(sentence):
    """Logs the edges and edge scores of a graph. 
    
    This information will be useful when reconstructing the graph after
    _Contract if a cycle is found.
    
    This is structured as a default dict, where each key is a head and each value is a list
    of tuples registering the candidate children and edge scores for the head, i.e. 
    {0: [(1, 9.0), (2, 10.0), (3, 9.0)]} etc.  
    
    Args: 
        sentence: sentence_pb2.Sentence(), the sentence to get edge information for.
    Returns:
        original_edges: defaultdict, dict of edges and edge scores.
    """
    original_edges = defaultdict(list)
    for token in sentence.token:
        # skip the ROOT token
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
            # the original selected_head is within the cycle, so skip it. 
            if candidate_head.address != token.selected_head.address:
                ch = new_token.candidate_head.add()
                ch.address = candidate_head.address
                ch.arc_score = candidate_head.arc_score - token.selected_head.arc_score + cycle_score
    #print(new_token)        

def _RedirectOutgoingArcs(cycle_tokens, outcycle_tokens, new_token):
    """Redirect any outgoing arcs from the cycle to the new target token. 
    
    For each token outside the cycle that has as head one of the tokens
    inside the cycle, craete an arc between the new_target to the outcycle
    token where the new target is the head of the outcycle token. This way
    we make sure that each token outside the cycle that has a head in the cycle
    now has the new_target as head. 
    
    
    Args:   
        cycle_tokens: the tokens inside the cycle.
        outcycle_tokens: the tokens outside the cycle. 
        new_token: the new target token. 
    """
    cycle_token_indexes = [token.index for token in cycle_tokens]
    for token in outcycle_tokens:
        # ignore if the token is ROOT.
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



def _GetCycleEdgesAndScores(original_edges, cycle_path):
    """Return cycle edges and edge scores for tokens in the cycle path.
    
    This function should only be called from _Reconstruct function. 
    
    Args:
        original_edges: defaultdict, the edges between the nodes of the original sentence.
        cycle_path: list, the path of the cycle.
    
    Returns:
        cycle_edges: dict, edges and edge_scores for nodes in the cycle. Useful during
            recontstruction.
    """
    cycle_edges = {}
    for source in set(cycle_path):
        targets = [target for target in original_edges[source] if target[0] in cycle_path]
        #print("cycle_path ", cycle_path)
        #print("targets: ", source, targets)
        cycle_target = max(targets, key=lambda x:x[1])
        #print("_GetCycleEdges cycle_targets: ", source, cycle_target)
        cycle_edges[source] = cycle_target
    return cycle_edges
                 

def _Reconstruct(cont_mst, new_token, original_edges, cycle_path):
    """Reconstruct.
    
    Args:
        cont_mst: sentence_pb2.maximum_spanning_tree, a contracted mst from which to reconstruct MST.
        new_token: sentence_pb2.Token, the new token that was added to the original sentence,
             to represent the cycle.
        original_edges: defaultdict, the edges between the nodes of the original sentence.
        cycle_path: list, the path of the cycle.
    
    Returns:
        reconstructed_edges: defaultdict, the edges of the original tree reconstructed after
            breaking the cycle. 
    """
    #logging.info("Reconstructing the edges for mst...")
    reconstructed_edges = defaultdict(list)
    cont_edges = defaultdict(list)
    cycle_edges = _GetCycleEdgesAndScores(original_edges, cycle_path)
    
    # a convenience function to find the edge score for a given source and target node index
    # from an edges dictionary; can be used to find edge score for a source and target in
    # original_edges or contracted edges. 
    get_edge_score = lambda source, target, edges_dict: [
        t[1] for t in edges_dict[source] if t[0] == target][0]
    
    # populate the defaultdict for contracted_edges.
    for token in cont_mst.sentence.token:
        if token.index == 0 or token.selected_head.address == -1:
            continue
        cont_edges[token.selected_head.address].append(
            (token.index, token.selected_head.arc_score)
        )
    
    #logging.info("Contracted sentence: {}".format(cont_sentence))
    logging.info("Original edges: {}".format(original_edges))
    logging.info("Contracted edges: {}".format(cont_edges))
    
    for source, target in cont_edges.items():
        
        # Handle arcs outgoing from the cycle.
        # logging.info("Handling arcs OUTGOING from the cycle.")
        if source == new_token.index:
            # This arc points from the cycle to outside. There might be more than one of these.
            # Find all the original nodes in the cycle that are responsible for such edges and
            # add them to the reconstructed_edges.
            cycle_target_index = cont_edges[source][0][0]
            cycle_target_score = cont_edges[source][0][1]
            #logging.info("target_edge_index: {}".format(cycle_target_index)) 
            #logging.info("target_edge_score: {}".format(cycle_target_score))

            # To find the original node, we look at the original_edges dict to understand which
            # node goes to the same target with the same arc_score in the original graph.
            for original_node_index in set(cycle_path):
                if get_edge_score(
                    original_node_index,
                    cycle_target_index,
                    original_edges) == cycle_target_score:
                    reconstructed_edges[original_node_index].append(
                        (cycle_target_index, cycle_target_score)
                    )
        
        # Handle arcs incoming to the cycle. 
        # logging.info("Handling arcs INCOMING to the cycle...")
        if target[0][0] == new_token.index:
            # This arc points at the cycle. Use the original_edges to find out which node
            # exactly in the cycle it points, then add all the edges in the cycle to the 
            # reconstructed_edges except for the one that completes the cycle loop. 
            original_target = max(original_edges[source], key=lambda x:x[1])
            reconstructed_edges[source].append(original_target)
            cycle_source = original_target[0]
            #logging.info("cycle_source: {}".format(cycle_source))
            cycle_target = cycle_edges[cycle_source]
            #logging.info("cycle_target: {}".format(cycle_target))
            while cycle_target[0] != original_target[0]:
                reconstructed_edges[cycle_source].append(cycle_target)
                cycle_source = cycle_target[0]
                #logging.info("cycle_source: {}".format(cycle_source))
                cycle_target = cycle_edges[cycle_source]
                #logging.info("cycle_target: {}".format(cycle_target))
    #logging.info("Reconstructed Edges: {}".format(reconstructed_edges))
    return reconstructed_edges


def _Merge(edges, sentence):
    """Merge the reconstructed edges into the original sentence.
    
    Args: 
        edges: defaultdict, the edges to merge into the sentence.
        sentence: the sentences which will be redone using the edges dict. 
    Returns:
        sentence
    """
    tokens = sentence.token
    for source, targets in edges.items():
        for token in tokens:
            for target in targets:
                if not token.index == target[0]:
                    continue
                logging.info("Determining head for token {} """.format(token.index))
                if token.selected_head.address != source:
                    logging.info("""Changing the head for {} from {} --> {}""".format(
                        token.index, token.selected_head.address, source
                        ))
                    logging.info("""Changing the arc score for {} from {} --> {}""".format(
                        token.index, token.selected_head.arc_score, target[1]
                        ))
                    token.selected_head.address = source
                    token.selected_head.arc_score = target[1]
                else:
                    logging.info("""Token {} already had correct head""".format(token.index))
    return _DropCandidateHeads(sentence)            


def _GetSentenceWeight(sentence):
    """Computes the total weight of the sentence."""
    weight = 0.0
    for token in sentence.token:
        weight += token.selected_head.arc_score
    return weight


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
    logging.info("""Input sentence is: {}""".format(sentence))
    graph = ConnectSentenceNodes(sentence)
    mst = ChuLiuEdmonds(graph)
    #sentence = sentence_pb2.Sentence()
    #with open(args.input_file, "rb") as sentence_proto:
    #    sentence.ParseFromString(sentence_proto.read())
    #    print(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to input file.")
    args = parser.parse_args()
    main(args)
