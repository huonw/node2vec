import random
from collections import defaultdict
import numpy as np
import gensim
from .categorical import Categorical
from joblib import Parallel, delayed
from tqdm import tqdm

from node2vec._native import ffi, lib

def parallel_generate_walks(probabilities, neighbors, global_walk_length, global_num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """
    walks = list()
    # Start the random walks from every node
    for source in tqdm(neighbors.keys(), desc='Generating walks (CPU: {})'.format(cpu_num)):
        # Calculate the number of walks, and how long, if there's a
        # specific strategy for this one.
        try:
            source_strategy = sampling_strategy[source]
        except KeyError:
            num_walks = global_num_walks
            walk_length = global_walk_length
        else:
            num_walks = source_strategy.get(num_walks_key, global_num_walks)
            walk_length = source_strategy.get(walk_length_key, global_walk_length)

        # Generate all the randomness we need up front, in one big
        # splat, for efficiency.
        rands = np.random.random((num_walks, walk_length - 1, 2))
        for walk_rands in rands:
            walk = [source]
            # Perform walk by stepping with each of those pairs of random
            # numbers
            for step_rands in walk_rands:
                previous = walk[-1]

                walk_options = neighbors[previous]

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    p = probabilities[previous]
                else:
                    p = probabilities[(walk[-2], previous)]

                idx = p.sample_with(step_rands[0], step_rands[1])
                walk_to = walk_options[idx]
                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    return walks

def parallel_precompute_probabilities(nodes, graph, sampling_strategy, global_p, global_q, cpu_num, weight_key, p_key, q_key):
    """
    Computes the transition probabilities for the random walks.
    :return: A pair of dictionaries: probabilities and neighbors.
    """

    # This maps id and (id, id) to an array of probabilities, where a
    # single id is set of first-travel probabilities (i.e. the transition
    # frequency when that id is the first in a walk), and a pair is
    # (source, current), i.e. the probabilities to use when an in-progress
    # walk is of the form [..., source, current].
    probabilities = {}
    neighbors = {}

    for source in tqdm(nodes, desc='Computing transition probabilities (CPU: %d)' % cpu_num):
        source_neighbors = graph[source]
        source_neighbors_set = set(source_neighbors)

        # Save the neighbors
        neighbors[source] = list(source_neighbors)

        first_travel_weights = list()

        for current_node in source_neighbors:
            try:
                sampling_strategy = sampling_strategy[current_node]
            except KeyError:
                p = global_p
                q = global_q
            else:
                p = sampling_strategy.get(P_KEY, global_p)
                q = sampling_strategy.get(Q_KEY, global_q)

            current_weight = source_neighbors[current_node].get(weight_key, 1)
            first_travel_weights.append(current_weight)

            unnormalized_weights = list()

            current_neighbors = graph[current_node]
            # Calculate unnormalized weights
            for destination in current_neighbors:
                raw_weight = current_neighbors[destination].get(weight_key, 1)
                if destination == source:  # Backwards probability
                    ss_weight = raw_weight / p
                elif destination in source_neighbors_set:  # If the neighbor is connected to the source
                    ss_weight = raw_weight
                else:
                    ss_weight = raw_weight / q

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnormalized_weights.append(ss_weight)

            if unnormalized_weights:
                probabilities[(source, current_node)] = Categorical(unnormalized_weights)

        if first_travel_weights:
            probabilities[source] = Categorical(first_travel_weights)

    return probabilities, neighbors

class Node2Vec:
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, probability_workers=None, sampling_strategy=None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.probability_workers = workers if probability_workers is None else probability_workers

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.probabilities, self.neighbors = self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        node_lists = np.array_split(list(self.graph.nodes), self.probability_workers)

        parallel = Parallel(n_jobs=self.probability_workers)
        results = parallel(delayed(parallel_precompute_probabilities)(nodes, self.graph,
                                                                      self.sampling_strategy, self.p, self.q,
                                                                      idx,
                                                                      self.WEIGHT_KEY, self.P_KEY, self.Q_KEY)
                           for idx, nodes in enumerate(node_lists, 1))

        probabilities = {}
        neighbors = {}
        for p, n in results:
            probabilities.update(p)
            neighbors.update(n)

        return probabilities, neighbors

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(delayed(parallel_generate_walks)(self.probabilities,
                                                                                      self.neighbors,
                                                                                      self.walk_length,
                                                                                      len(num_walks),
                                                                                      idx,
                                                                                      self.sampling_strategy,
                                                                                      self.NUM_WALKS_KEY,
                                                                                      self.WALK_LENGTH_KEY) for
                                                     idx, num_walks
                                                     in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
