import networkx as nx
import numpy as np
from .node2vec import Node2Vec

# The preprocessed probabilities are a non-random property of the
# graph, so we can test that we get them right.

def assert_probabilities_equal(computed, expected):
    """Checks that the Node2Vec computed probabilities are what's
    expected."""
    assert computed.keys() == expected.keys()
    for k in expected:
        np.testing.assert_almost_equal(computed[k].pmf(), expected[k],
                                       err_msg='mismatched probabilities for %s' % (k,))

def n(array):
    """Normalize the array to sum to 1."""
    x = np.array(array)
    return x / x.sum()

# search bias parameters for returning (d_tx = 0), triangles (d_tx =
# 1) and exploring (d_tx = 2). The parameters we control are all
# distinct primes so that we can be sure that the multiplications are
# including the right values.
RET = 2
TRI = 1
EXP = 3

def test_unweighted_probabilities():
    # 0-1-2-4
    #   |/
    #   3
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1),
         (1, 2),
         (2, 3),
         (1, 3),
         (2, 4)])

    n2v = Node2Vec(graph, dimensions=1, walk_length=1, num_walks=1, p=1/RET, q=1/EXP)

    assert n2v.neighbors == {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3, 4],
        3: [2, 1],
        4: [2],
    }

    assert_probabilities_equal(n2v.probabilities, {
        # first, the first-travel weights (all equal in an unweighted
        # graph), i.e. the walk is just [current]
        0: n([1]),
        1: n([1, 1, 1]),
        2: n([1, 1, 1]),
        3: n([1, 1]),
        4: n([1]),

        # second, the 2-hop biased probabilities, i.e. the walk is
        # [..., source, current]

        # current == 0
        (1, 0): n([RET]),

        # current == 1
        (0, 1): n([RET, EXP, EXP]),
        (2, 1): n([EXP, RET, TRI]),
        (3, 1): n([EXP, TRI, RET]),

        # current == 2
        (1, 2): n([RET, TRI, EXP]),
        (3, 2): n([TRI, RET, EXP]),
        (4, 2): n([EXP, EXP, RET]),

        # current == 3
        (1, 3): n([TRI, RET]),
        (2, 3): n([RET, TRI]),

        # current == 4
        (2, 4): n([2])
    })

def test_weighted_probabilities():
    # 0-1-2
    #   |/
    #   3
    graph = nx.Graph()
    w01 = 5
    w12 = 7
    w13 = 11
    w23 = 13
    graph.add_edges_from(
        [(0, 1, { Node2Vec.WEIGHT_KEY: w01 }),
         (1, 2, { Node2Vec.WEIGHT_KEY: w12 }),
         (1, 3, { Node2Vec.WEIGHT_KEY: w13 }),
         (2, 3, { Node2Vec.WEIGHT_KEY: w23 })])

    n2v = Node2Vec(graph, dimensions=1, walk_length=1, num_walks=1, p=1/RET, q=1/EXP)

    assert n2v.neighbors == {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3],
        3: [1, 2]
    }

    assert_probabilities_equal(n2v.probabilities, {
        # the first-travel probabilities
        0: n([w01]),
        1: n([w01, w12, w13]),
        2: n([w12, w23]),
        3: n([w13, w23]),

        # the two-hop biased probabilities

        # current == 0
        (1, 0): n([RET * w01]),

        # current == 1
        (0, 1): n([RET * w01, EXP * w12, EXP * w13]),
        (2, 1): n([EXP * w01, RET * w12, TRI * w13]),
        (3, 1): n([EXP * w01, TRI * w12, RET * w13]),

        # current == 2
        (1, 2): n([RET * w12, TRI * w23]),
        (3, 2): n([TRI * w12, RET * w23]),

        # current == 3
        (1, 3): n([RET * w13, TRI * w23]),
        (2, 3): n([TRI * w13, RET * w23])
    })
