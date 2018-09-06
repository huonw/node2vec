from .categorical import Categorical
import numpy as np

N = 10000
def check_pmf(c, expected):
    np.testing.assert_almost_equal(c.pmf(), expected)

    observed = np.bincount([c.sample() for _ in range(N)])
    np.testing.assert_almost_equal(observed / observed.sum(), expected,
                                   decimal=2)

def test_uniform1():
    c = Categorical([1234])
    np.testing.assert_almost_equal(c.weights, [1])
    check_pmf(c, [1] * 1)

def test_uniform2():
    c = Categorical([1234] * 2)
    np.testing.assert_almost_equal(c.weights, [1] * 2)

    check_pmf(c, [1/2] * 2)

def test_uniform100():
    c = Categorical([1234] * 100)
    np.testing.assert_almost_equal(c.weights, [1] * 100)

    check_pmf(c, [1/100] * 100)

def test_nonuniform2():
    c = Categorical([1, 2])
    np.testing.assert_almost_equal(c.weights, [2/3, 1])
    np.testing.assert_almost_equal(c.reassigns, [1, 1])

    check_pmf(c, [1/3, 2/3])

def test_nonuniform100():
    c = Categorical(list(range(1, 100 + 1)))
    # it's too hard to work out the exact sequence of assignments
    check_pmf(c, np.arange(1, 101) / 5050)

def test_nonuniform100_shuffle():
    rng = np.arange(0, 100)
    np.random.shuffle(rng)
    c = Categorical(rng)
    check_pmf(c, rng / rng.sum())


