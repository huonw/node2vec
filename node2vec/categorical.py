# The MIT License (MIT)
# Copyright (c) 2016 Edward Newell

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
Enables fast on-demand sampling from a categorical probability distribution

Adopted and adjusted from: https://github.com/enewe101/categorical
'''

import numpy.random as r
import numpy as np


class Categorical(object):

    def __init__(self, scores):
        if len(scores) < 1:
            raise ValueError('The scores list must have length >= 1')

        k = len(scores)
        weights = np.array(scores, dtype=np.float64)
        weights *= k / weights.sum()
        reassigns = np.arange(k, dtype=np.int32)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = list(np.where(weights < 1.0)[0])
        larger = list(np.where(weights > 1.0)[0])

        # We will have k different slots. Each slot represents 1/K
        # prbability mass, and to each we allocate all of the probability
        # mass from a "small" outcome, plus some probability mass from
        # a "large" outcome (enough to equal the total 1/K).
        # We keep track of the remaining mass left for the larger outcome,
        # allocating the remainder to another slot later.
        # The result is that the kth has some mass allocated to outcome
        # k, and some allocated to another outcome, recorded in J[k].
        # q[k] keeps track of how much mass belongs to outcome k, and 
        # how much belongs to outcome J[k].
        while smaller and larger:
            small_idx = smaller.pop()
            large_idx = larger.pop()
 
            reassigns[small_idx] = large_idx
            weights[large_idx] -= 1.0 - weights[small_idx]

            if weights[large_idx] < 1.0:
                smaller.append(large_idx)
            else:
                larger.append(large_idx)

        # set things to one, in case of floating point error
        while smaller:
            idx = smaller.pop()
            weights[idx] = 1.0
        while larger:
            idx = larger.pop()
            weights[idx] = 1.0

        self.weights = weights
        self.reassigns = reassigns

    def probability(self, k):
        '''
        Get the actual probability associated to outcome k. O(self.size())
        '''
        size = len(self.weights)
        if not (0 <= k < size):
            return 0
        return self.pmf()[k]

    def pmf(self):
        '''
        Get the probabilities associated with each outcome. O(self.size())
        '''
        pmf = self.weights.copy()
        ints = self.reassigns.astype(np.int64)
        np.add.at(pmf, ints, 1)
        np.subtract.at(pmf, ints, self.weights)
        pmf /= len(pmf)
        return pmf

    def sample(self):
        # Draw from the overall uniform mixture.
        k = np.random.choice(len(self.weights))

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        u = self.weights[k]
        if u == 1:
            return int(k)
        elif np.random.uniform() < u:
            return int(k)
        else:
            return int(self.reassigns[k])
