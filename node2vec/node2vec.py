import random
from collections import defaultdict
import numpy as np
import gensim
from .categorical import Categorical
from joblib import Parallel, delayed
from tqdm import tqdm

def parallel_generate_walks(probabilities, neighbors, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """
    walks = list()
    with tqdm(total=num_walks * len(neighbors)) as pbar:
        pbar.set_description('Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):

            # Shuffle the nodes
            shuffled_nodes = list(neighbors.keys())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
                pbar.update(1)

                # Skip nodes with specific num_walks
                if source in sampling_strategy and \
                        num_walks_key in sampling_strategy[source] and \
                        sampling_strategy[source][num_walks_key] <= n_walk:
                    continue

                # Start walk
                walk = [source]

                # Calculate walk length
                if source in sampling_strategy:
                    walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
                else:
                    walk_length = global_walk_length

                # Perform walk
                while len(walk) < walk_length:
                    previous = walk[-1]

                    walk_options = neighbors[previous]

                    # Skip dead end nodes
                    if not walk_options:
                        break

                    if len(walk) == 1:  # For the first step
                        p = probabilities[previous]
                    else:
                        p = probabilities[(walk[-2], previous)]

                    idx = p.sample()
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

            probabilities[(source, current_node)] = Categorical(unnormalized_weights)
        probabilities[source] = Categorical(first_travel_weights)

    return probabilities, neighbors

class Node2Vec:
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None):
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

        node_lists = np.array_split(list(self.graph.nodes), self.workers)

        results = Parallel(n_jobs=self.workers)(delayed(parallel_precompute_probabilities)(nodes, self.graph,
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
