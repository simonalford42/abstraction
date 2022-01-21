from ctypes.wintypes import tagRECT
import math
from utils import assertEqual
from scipy.special import logsumexp
import unittest
import numpy as np


def forward(init_dist, trans_fn, obs_dist, obs):
    """
    observations are in [0, a-1]
    states are in [0, b-1]
    init_dist: (b, ) vector of initial distribution over states
    trans_fn: (b, b) matrix whose i,j'th entry is probability of transitioning
        from state i to state j obs_dist: (b, a) matrix whose i,j'th entry is
        probability of outputing observation j given state i.
    obs: (T, ) observation of states.

    returns (matrix, total_prob), where the t,i'th entry of matrix is the
    probability of the observation up to time t given the t'th state is i.
    """
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[0] = init_dist * p_obs_at_t_given_state[0, :]

    for t in range(1, T):
        # new_f[j] = f[i] * trans_fn[i, j] * p_obs_at_t_given_state[t, j]
        f_matrix[t] = np.einsum('i, ij, j -> j', f_matrix[t-1], trans_fn, p_obs_at_t_given_state[t])

    p = f_matrix[T-1].sum()
    return f_matrix, p


def forward_log(init_dist, trans_fn, obs_dist, obs):
    """
    observations are in [0, a-1]
    states are in [0, b-1]
    init_dist: (b, ) vector of initial distribution over states
    trans_fn: (b, b) matrix whose i,j'th entry is probability of transitioning
        from state i to state j obs_dist: (b, a) matrix whose i,j'th entry is
        probability of outputing observation j given state i.
    obs: (T, ) observation of states.

    returns (matrix, total_prob), where the t,i'th entry of matrix is the
    probability of the observation up to time t given the t'th state is i.
    """
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    trans_fn = np.log(trans_fn)
    obs_dist = np.log(obs_dist)
    init_dist = np.log(init_dist)

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[0] = init_dist + p_obs_at_t_given_state[0, :]

    for t in range(1, T):
        # new_f[j] = f[i] * trans_fn[i, j] * p_obs_at_t_given_state[t, j]
        new_f_vector = (f_matrix[t-1])[:, np.newaxis] + trans_fn + (p_obs_at_t_given_state[t])[np.newaxis, :]  # (a, a)
        f_matrix[t] = logsumexp(new_f_vector, axis=0)

    p = logsumexp(f_matrix[T-1], axis=0)
    return f_matrix, p


def backward(init_dist, trans_fn, obs_dist, obs):
    """
    observations are in [0, a-1]
    states are in [0, b-1]
    init_dist: (b, ) vector of initial distribution over states
    trans_fn: (b, b) matrix whose i,j'th entry is probability of transitioning
        from state i to state j obs_dist: (b, a) matrix whose i,j'th entry is
        probability of outputing observation j given state i.
    obs: (T, ) observation of states.

    returns (matrix, total_prob), where the t,i'th entry of matrix is the
    probability of the observation from t+1 onwards given the t'th state is i.
    and total_prob is the total probability
    """
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[-1] = np.ones(b)

    for t in range(T-2, -1, -1):
        # new_f[i] = f[j] * trans_fn[i, j] * p_obs_at_t_given_state[t, j]
        f_matrix[t] = np.einsum('j, ij, j -> i', f_matrix[t+1], trans_fn, p_obs_at_t_given_state[t+1])

    p = (f_matrix[0] * init_dist * p_obs_at_t_given_state[0]).sum()
    return f_matrix, p


def backward_log(init_dist, trans_fn, obs_dist, obs):
    """
    observations are in [0, a-1]
    states are in [0, b-1]
    init_dist: (b, ) vector of initial distribution over states
    trans_fn: (b, b) matrix whose i,j'th entry is probability of transitioning
        from state i to state j obs_dist: (b, a) matrix whose i,j'th entry is
        probability of outputing observation j given state i.
    obs: (T, ) observation of states.

    returns (matrix, total_prob), where the t,i'th entry of matrix is the
    log probability of the observation from t+1 onwards given the t'th state is i.
    """
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    trans_fn = np.log(trans_fn)
    obs_dist = np.log(obs_dist)
    init_dist = np.log(init_dist)

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[-1] = np.zeros(b)

    for t in range(T-2, -1, -1):
        # new_f[i] = f[j] * trans_fn[i, j] * p_obs_at_t_given_state[t, j]
        new_f = (f_matrix[t+1])[np.newaxis, :] + trans_fn + (p_obs_at_t_given_state[t+1])[np.newaxis, :]
        f_matrix[t] = logsumexp(new_f, axis=1)

    p = logsumexp(f_matrix[0] + init_dist + p_obs_at_t_given_state[0])
    return f_matrix, p


def viterbi(init_dist, trans_fn, obs_dist, obs):
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[0] = init_dist * p_obs_at_t_given_state[0]
    pointer_matrix = np.zeros((T, b))

    for t in range(1, T):
        f_new = (f_matrix[t-1])[:, np.newaxis] * trans_fn
        pointers = np.argmax(f_new, axis=0)

        f_matrix[t] = p_obs_at_t_given_state[t] * np.max(f_new, axis=0)
        pointer_matrix[t] = pointers

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(f_matrix[-1])
    for t, pointers in zip(range(T-2, -1, -1), pointer_matrix[::-1]):
        path[t] = int(pointers[path[t+1]])

    return path


def fw_bw(init_dist, trans_fn, obs_dist, obs):
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    # see derivation at https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Update
    alpha, _ = forward(init_dist, trans_fn, obs_dist, obs)
    beta, _ = backward(init_dist, trans_fn, obs_dist, obs)
    p_obss = np.sum(alpha * beta, axis=1)
    # calculation should be the same for each point in time
    assert np.all(p_obss[0] == p_obss), 'mistake in fw/bw calculations'
    p_obs = p_obss[0]

    gamma = (alpha * beta) / p_obs
    assertEqual(gamma.shape, (T, b))
    xi_numerator = np.einsum('ti, ij, tj, tj -> tij', alpha[:-1], trans_fn, p_obs_at_t_given_state[1:], beta[1:])
    assertEqual(xi_numerator.shape, (T-1, b, b))
    xi = xi_numerator / p_obs

    new_trans_fn_denom = np.sum(gamma[:-1], axis=0)
    alt_new_trans_fn_denom = np.einsum('tij->i', xi)
    np.testing.assert_allclose(new_trans_fn_denom, alt_new_trans_fn_denom)

    new_init_dist = gamma[0]
    new_trans_fn = np.sum(xi, axis=0) / new_trans_fn_denom[:, np.newaxis]

    obs_indicator = (obs == np.arange(a)[:, np.newaxis]).transpose()
    assertEqual(obs_indicator.shape, (T, a))

    new_obs_dist = np.einsum('ta, tb -> ba', obs_indicator, gamma) / np.einsum('ti -> i', gamma)[:, np.newaxis]
    assertEqual(new_obs_dist.shape, (b, a))

    np.testing.assert_almost_equal(np.sum(new_init_dist, axis=0), 1)
    np.testing.assert_almost_equal(np.sum(new_trans_fn, axis=1), np.ones(b))
    np.testing.assert_almost_equal(np.sum(new_obs_dist, axis=1), np.ones(b))

    return new_init_dist, new_trans_fn, new_obs_dist, p_obs


def viterbi_log(init_dist, trans_fn, obs_dist, obs):
    (b, ) = init_dist.shape
    assertEqual(trans_fn.shape, (b, b))
    assertEqual(obs_dist.shape[0], b)
    a = obs_dist.shape[1]
    (T, ) = obs.shape

    assertEqual(sum(init_dist), 1)
    assertEqual(np.sum(trans_fn, axis=1), np.ones(b))
    assertEqual(np.sum(obs_dist, axis=1), np.ones(b))

    # (T, b) matrix whose i,j'th entry is probability of observation from time step i given state j.
    p_obs_at_t_given_state = obs_dist[:, obs].transpose()

    f_matrix = np.zeros((T, b))
    f_matrix[0] = init_dist + p_obs_at_t_given_state[0]
    pointer_matrix = np.zeros((T, b))

    for t in range(1, T):
        f_new = (f_matrix[t-1])[:, np.newaxis] + trans_fn
        pointers = np.argmax(f_new, axis=0)

        f_matrix[t] = p_obs_at_t_given_state[t] + np.max(f_new, axis=0)
        pointer_matrix[t] = pointers

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(f_matrix[-1])
    for t, pointers in zip(range(T-2, -1, -1), pointer_matrix[::-1]):
        path[t] = int(pointers[path[t+1]])

    return path


class HMMTest(unittest.TestCase):

    def check_fw_bw(self, fw_probs, bw_probs):
        obs_probs = np.einsum('tj, tj -> t', fw_probs, bw_probs)
        self.assertTrue(np.all(obs_probs == obs_probs[0]))

    def test1(self):
        # b = 2, a = 2
        trans_fn = np.array([[1, 0], [0.75, 0.25]])
        obs_dist = np.array([[0.75, 0.25], [0, 1]])

        obs = np.array([1, 0, 0])
        init_dist = np.array([1, 0])
        fw_out, p = forward(init_dist, trans_fn, obs_dist, obs)
        target = np.array([[0.25, 0],
                           [0.25 * 0.75, 0],
                           [0.25 * 0.75 * 0.75, 0]])
        target_p = 0.25 * 0.75 * 0.75

        self.assertTrue(np.array_equal(fw_out, target))
        self.assertEqual(p, target_p)

        out2, p2 = forward_log(init_dist, trans_fn, obs_dist, obs)
        self.assertTrue(np.allclose(out2, np.log(fw_out)))
        self.assertTrue(math.isclose(p2, np.log(p)))

        bw_out, p = backward(init_dist, trans_fn, obs_dist, obs)
        target = np.array([[1, 1], [0.75, 0.75 * 0.75], [0.75 * 0.75, 0.75 ** 3]])
        target = target[::-1]  # I entered them backwards
        # target_p stays the same

        np.testing.assert_array_equal(bw_out, target)
        self.assertTrue(np.array_equal(bw_out, target))
        self.assertEqual(p, target_p)

        out2, p2 = backward_log(init_dist, trans_fn, obs_dist, obs)
        np.testing.assert_allclose(out2, np.log(bw_out))
        self.assertTrue(np.allclose(out2, np.log(bw_out)))
        self.assertTrue(math.isclose(p2, np.log(p)))

        self.check_fw_bw(fw_out, bw_out)

        most_likely = [0, 0, 0]
        v = viterbi(init_dist, trans_fn, obs_dist, obs)
        v2 = viterbi_log(init_dist, trans_fn, obs_dist, obs)
        self.assertEqual(list(v), most_likely)
        self.assertEqual(list(v2), most_likely)

    def test2(self):
        # b = 2, a = 2
        trans_fn = np.array([[1, 0], [0.75, 0.25]])
        obs_dist = np.array([[0.75, 0.25], [0, 1]])

        obs = np.array([1, 0])
        init_dist = np.array([0, 1])
        fw_out, p = forward(init_dist, trans_fn, obs_dist, obs)
        target = np.array([[0, 1], [0.75 * 0.75, 0]])
        target_p = 0.75 * 0.75

        self.assertTrue(np.array_equal(fw_out, target))
        self.assertEqual(p, target_p)

        out2, p2 = forward_log(init_dist, trans_fn, obs_dist, obs)
        self.assertTrue(np.allclose(out2, np.log(fw_out)))
        self.assertTrue(math.isclose(p2, np.log(p)))

        bw_out, _ = backward(init_dist, trans_fn, obs_dist, obs)
        self.check_fw_bw(fw_out, bw_out)

        most_likely = [1, 0]
        v = viterbi(init_dist, trans_fn, obs_dist, obs)
        v2 = viterbi_log(init_dist, trans_fn, obs_dist, obs)
        self.assertEqual(list(v), most_likely)
        self.assertEqual(list(v2), most_likely)

    def test3(self):
        # b = 2, a = 2
        trans_fn = np.array([[1, 0], [0.75, 0.25]])
        obs_dist = np.array([[0.75, 0.25], [0, 1]])

        obs = np.array([0, 0])
        init_dist = np.array([0.5, 0.5])
        fw_out, p = forward(init_dist, trans_fn, obs_dist, obs)

        target = np.array([[0.5 * 0.75, 0], [0.5 * 0.75 * 0.75, 0]])
        target_p = 0.5 * 0.75 * 0.75

        self.assertTrue(np.array_equal(fw_out, target))
        self.assertEqual(p, target_p)

        bw_out, _ = backward(init_dist, trans_fn, obs_dist, obs)
        self.check_fw_bw(fw_out, bw_out)

        out2, p2 = forward_log(init_dist, trans_fn, obs_dist, obs)
        self.assertTrue(np.allclose(out2, np.log(fw_out)))
        self.assertTrue(math.isclose(p2, np.log(p)))

        most_likely = [0, 0]
        v = viterbi(init_dist, trans_fn, obs_dist, obs)
        v2 = viterbi_log(init_dist, trans_fn, obs_dist, obs)
        self.assertEqual(list(v), most_likely)
        self.assertEqual(list(v2), most_likely)

    def test4(self):
        trans_fn = np.array([[0.6, 0.4], [0.3, 0.7]])
        obs_dist = np.array([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])

        obs = np.array([0, 1, 2, 2])
        init_dist = np.array([0.8, 0.2])
        fw_out, p = forward(init_dist, trans_fn, obs_dist, obs)

        target = np.array([[0.24, 0.08], [0.067, 0.046], [0.016, 0.017], [0.0045, 0.0056]])
        target_p = 0.0045 + 0.0056

        self.assertTrue(np.allclose(fw_out, target, atol=0.005))
        self.assertTrue(math.isclose(p, target_p, abs_tol=0.005))

        out2, p2 = forward_log(init_dist, trans_fn, obs_dist, obs)
        self.assertTrue(np.allclose(out2, np.log(fw_out)))
        self.assertTrue(math.isclose(p2, np.log(p), abs_tol=0.005))

        bw_out, p = backward(init_dist, trans_fn, obs_dist, obs)
        target = np.array([[1, 1], [0.3, 0.3], [0.09, 0.09], [0.0324, 0.0297]])
        target = target[::-1]  # I entered them backwards
        # target_p stays the same

        self.check_fw_bw(fw_out, bw_out)

        np.testing.assert_array_equal(bw_out, target)
        self.assertTrue(np.array_equal(bw_out, target))
        self.assertTrue(math.isclose(p, target_p, abs_tol=0.005))

        out2, p2 = backward_log(init_dist, trans_fn, obs_dist, obs)
        np.testing.assert_allclose(out2, np.log(bw_out))
        self.assertTrue(np.allclose(out2, np.log(bw_out)))
        self.assertTrue(math.isclose(p2, np.log(p)))

        most_likely = [0, 0, 0, 0]
        v = viterbi(init_dist, trans_fn, obs_dist, obs)
        v2 = viterbi_log(init_dist, trans_fn, obs_dist, obs)
        self.assertEqual(list(v), most_likely)
        self.assertEqual(list(v2), most_likely)

        fw_out, p = forward(init_dist, trans_fn, obs_dist, obs)
        bw_out, p2 = backward(init_dist, trans_fn, obs_dist, obs)

        new_init_dist, new_trans_fn, new_obs_dist, p = fw_bw(init_dist, trans_fn, obs_dist, obs)
        target_new_trans_fn = np.array([[0.63, 0.37], [0.31, 0.69]])
        np.testing.assert_array_almost_equal(new_trans_fn, target_new_trans_fn, decimal=2)



if __name__ == '__main__':
    unittest.main()
