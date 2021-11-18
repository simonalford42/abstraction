import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset  # , DataLoader
import utils
from utils import assertEqual, num_params, FC
from torch.distributions import Categorical


class AbstractController(nn.Module):
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        self.b = n_abstractions
        self.s = state_dim
        self.n = n_micro_actions
        self.micro_net = FC(input_dim=self.s,
                            output_dim=self.b * self.n,
                            num_hidden=0)
        self.stop_net = FC(input_dim=self.s,
                           output_dim=self.b * 2,
                           num_hidden=2,
                           hidden_dim=64)
        self.start_net = FC(input_dim=self.s,
                            output_dim=self.b,
                            num_hidden=2,
                            hidden_dim=64)

    def forward(self, state_embeds):
        """
        input: (T, s) tensor of states
        outputs:
           (T, b, n) tensor of logps
           (T, b, 2) tensor of stop logits
               interpretation: (T, b, 0) is "keep going" (stop=False) (1 - beta)
                               (T, b, 1) is stop logp
           (T, b) tensor of start logits
        so intermediate out (T, num_abstractions * (n + 2))
        """
        T = state_embeds.shape[0]
        assertEqual(state_embeds.shape, (T, self.s))

        action_out = self.micro_net(state_embeds)
        stop_out = self.stop_net(state_embeds)
        start_logits = self.start_net(state_embeds)

        action_logits = action_out.reshape(T, self.b, self.n)
        stop_logits = stop_out.reshape(T, self.b, 2)

        action_logits = F.log_softmax(action_logits, dim=2)
        stop_logits = F.log_softmax(stop_logits, dim=2)
        start_logits = F.log_softmax(start_logits, dim=1)

        return action_logits, stop_logits, start_logits


class Eq2Net(nn.Module):
    def __init__(self, n_abstractions, state_dim, n_micro_actions,
                 abstract_penalty=0.5, model='DP'):
        """
        DP: dynamic programming model
        HMM: HMM model
        micro: no options.
        """
        super().__init__()
        self.b = n_abstractions
        self.s = state_dim
        self.n = n_micro_actions
        self.controller = AbstractController(n_abstractions,
                                             state_dim,
                                             n_micro_actions)
        # logp penalty for longer sequences
        self.abstract_penalty = abstract_penalty
        assert model in ['DP', 'HMM', 'micro']
        self.model = model

    def forward(self, state_embeds, actions):
        if self.model == 'DP':
            return self.forward_DP(state_embeds, actions)
        elif self.model == 'HMM':
            return self.forward_HMM(state_embeds, actions)
        elif self.model == 'micro':
            return self.forward_micro(state_embeds, actions)
        else:
            assert False

    def forward_DP(self, state_embeds, actions):
        """
        state_embeds: (T+1, s) tensor
        actions: (T, ) tensor of ints

        outputs: logp of whole sequence
        """
        T = len(actions)
        assert state_embeds.shape == (T + 1, self.s)
        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.controller(state_embeds)

        # (T, b, n) this step is automatically done by next line
        # action_logits = action_logits[:-1, ::]

        # don't need logps for actions for last state, which is where we stop
        # see https://numpy.org/doc/stable/reference/arrays.indexing.html#purely-integer-array-indexing
        # extracts logps for actions chosen for each timestep, keeping b axis
        step_logps = action_logps[range(T), :, actions]
        assertEqual(step_logps.shape, (T, self.b))

        def logp_subseq(i, j):
            """ seq is (j - i - 1) actions long, so ends at j """
            #               prob of option being initialized at s_0
            logp_given_b = (start_logps[i, :]
                            # prob of actions along sequence
                            + step_logps[i:j, :].sum(axis=0)
                            # prob of not stopping before end; assumes len(option) > 1
                            + stop_logps[i+1:j, :, 0].sum(axis=0)
                            # prob of stopping at end
                            + stop_logps[j, :, 1])
            assertEqual(logp_given_b.shape, (self.b, ))
            return torch.logsumexp(logp_given_b, dim=0)

        # last state has nowhere to go, so parses to logp 0.
        # second to last state has one option, so recursion starts there
        logp_table = [0] * (T + 1)

        # first seq to calculate is (i=T-1, j=T) -- a length one option at end
        # last is the full sequence (i=0, j=T)
        for i in range(T - 1, -1, -1):
            logps = torch.stack([logp_subseq(i, j) + logp_table[j]
                                 # (a, a + 1) has two states, one action
                                 # (a, T) goes to very end
                                 for j in range(i + 1, T + 1)])
            # sum over options
            logp = torch.logsumexp(logps, dim=0)
            # don't penalize last step with no split later.
            if i > 0:
                # note: -= doesn't work!
                logp = logp - self.abstract_penalty
            logp_table[i] = logp

        return logp_table[0]

    def forward_HMM(self, state_embeds, actions):
        """
        state_embeds: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: logp of sequence

        HMM calculation, identical to Smith et al. 2018.
        """
        T = len(actions)
        assertEqual(state_embeds.shape, (T + 1, self.s))
        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.controller(state_embeds)

        total_logp = 0.
        # log-softmaxed to one.
        macro_dist = start_logps[0]  # [b]
        for i, action in enumerate(actions):
            # invariant: prob dist should sum to 1
            # I was getting error of ~1E-7 which got triggered by default value
            # only applies if no abstract penalty
            if not self.abstract_penalty:
                assert torch.isclose(torch.logsumexp(macro_dist, dim=0),
                                     torch.tensor(0.), atol=1E-5), \
                       f'Not quite zero: {torch.logsumexp(macro_dist, dim=0)}'

            # print(f"macro_dist: {macro_dist}")

            # transition before acting. this way the state at which an option
            # starts is where its first move happens
            # => skip transition for the first step
            if i > 0:
                # markov transition, mirroring policy over options from Smith
                stop_lps = stop_logps[i, :, 1]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, 0]  # (b,)

                start_lps = start_logps[i]  # (b,)
                # how much probability mass exits each option
                macro_stops = macro_dist + stop_lps
                total_rearrange = torch.logsumexp(macro_stops, dim=0)
                total_rearrange = total_rearrange - self.abstract_penalty
                # distribute new mass among new options
                new_mass = start_lps + total_rearrange
                # mass that stays in place, aka doesn't stop
                macro_dist = macro_dist + one_minus_stop_lps
                # add new mass
                macro_dist = torch.logaddexp(macro_dist, new_mass)

            # (b,)
            action_lps = action_logps[i, :, action]
            # in prob space, this is a sum of probs weighted by macro-dist
            logp = torch.logsumexp(action_lps + macro_dist, dim=0)
            total_logp += logp

        # all macro options need to stop at the very end.
        final_stop_lps = stop_logps[-1, :, 0]
        total_logp += torch.logsumexp(final_stop_lps + macro_dist, dim=0)

        return total_logp

    def forward_micro(self, state_embeds, actions):
        """
        state_embeds: (T+1, state_dim) tensor
        actions: (T, ) tensor of ints

        outputs: logp of sequence
        """
        T = len(actions)
        assert state_embeds.shape == (T + 1, self.s)
        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.controller(state_embeds)

        step_logps = torch.stack([action_logps[i, 0, actions[i]]
                                 for i in range(len(actions))])
        return step_logps.sum(axis=0)


def generate_grid_traj(scale, n):
    # random goal which is n up/right steps away
    x = random.randint(0, n)
    y = n - x

    traj = ['R'] * x + ['U'] * y
    random.shuffle(traj)
    micro_traj = [m * scale for m in traj]
    # go from list to single string
    micro_traj = ''.join(micro_traj)

    return micro_traj


def exec_traj(traj):
    """
    input: ['U', 'U', 'R', 'R']
    output: [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
    """
    x, y = (0, 0)
    trace = [(x, y)]

    for move in traj:
        assert move in ['U', 'R'], f'{move} is not U or R'
        if move == 'U':
            y += 1
        else:
            x += 1

        trace.append((x, y))

    return trace


def generate_data(scale, seq_len, n):
    return [generate_grid_traj(scale, seq_len) for _ in range(n)]


class TrajData(Dataset):
    def __init__(self, trajs, max_coord=None):
        """
        trajs: list of trajectories a la generate_grid_traj()

        generates ((x, y, x_goal, y_goal), move) data points from the trajs, and
        makes a dataset of them.

        Note: for trajectories in sequence form, we include the final state
        reached as a tuple (state, None) i.e. no action/move for that tuple.
        These are not included as singleton data points for policy net
        training.

        0 = right = 'R'
        1 = up = 'U'
        """
        super().__init__()
        # list of ['U','U','R','R',...]
        self.trajs = trajs
        self.seq_len = len(self.trajs[0])
        # not necessary, but currently assuming in code elsewhere
        assert all(len(t) == self.seq_len for t in self.trajs)

        # list of list of ((x, y, x_goal, y_goal), move) tuples
        self.points_lists = [TrajData.make_points(t) for t in trajs]
        # list of ((x, y, x_goal, y_goal), move) tuples
        self.points = [p for points in self.points_lists
                       for p in points if p[1] != -1]

        # list of (x, y, x_goal, y_goal)
        # list of moves (int)
        self.coords, self.moves = zip(*self.points)
        self.max_coord = max(max(c) for c in self.coords)
        print(f"self.max_coord: {self.max_coord}")
        if max_coord:
            assert max_coord >= self.max_coord, 'invalid manual max_coord'
            self.max_coord = max_coord

        self.state_dim = 4 * (self.max_coord + 1)

        # list of list of (state_embed, move)
        # includes (goal, -1) at end
        self.traj_embeds = [self.embed_points(p)
                            for p in self.points_lists]
        # list of list of state_embed
        self.traj_states = [[s for (s, m) in traj_embed]
                            for traj_embed in self.traj_embeds]
        # list of tensor of moves
        # does not include -1 at end
        self.traj_moves = [torch.tensor([m for (s, m) in traj_embed[:-1]])
                           for traj_embed in self.traj_embeds]

        # list of state_embeds.
        self.state_embeds = [s for traj_embed in self.traj_embeds
                             for s, m in traj_embed if m != -1]

        # each trajectory as a (traj_length, state_dim) tensor
        self.traj_batches = [torch.stack([s for (s, m) in traj_embed])
                             for traj_embed in self.traj_embeds]

    def make_points(traj):
        trace = exec_traj(traj)
        assertEqual(len(trace), len(traj) + 1)
        goal = trace[-1]

        # list of ((x, y, x_goal, y_goal), move) tuples
        points = [((*point, *goal),
                   torch.tensor(0 if move == 'R' else 1))
                  for point, move in zip(trace[:-1], traj)]
        # needed so we can probe probability of stopping at the end!
        points.append(((*goal, *goal), torch.tensor(-1)))
        return points

    def execute(point, action):
        (x, y) = point
        if action == 0:
            return (x + 1, y)
        else:
            return (x, y + 1)

    def embed_state(self, s):
        """
        from (x, y, x_goal, y_goal) tensor
        to concated one-hot tensor of shape 4 * (max_coord + 1)
        """
        s = torch.cat([torch.tensor([coord]) for coord in s])
        s = F.one_hot(s, num_classes=self.max_coord + 1).to(torch.float)
        assertEqual(s.shape, (4, self.max_coord + 1))
        s = s.flatten()
        assertEqual(s.shape, (4 * (self.max_coord + 1), ))
        return s

    def embed_points(self, points):
        # from list of ((x, y, x_gaol, y_goal), move)
        # to list of (state_embed, move)
        return [(self.embed_state(state), move) for state, move in points]

    def convert_traj(self, traj):
        # convert traj from UURRUU to list of (state_embed, action)
        points = self.make_points(traj)
        return self.embed_points(points)

    def __len__(self):
        return len(self.state_embeds)

    def __getitem__(self, idx):
        return (self.state_embeds[idx],
                self.moves[idx])


def train_abstractions(data, net, epochs, lr=1E-3):
    print(f"net has {num_params(net)} parameters")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = []
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for state_embeds, actions in zip(data.traj_batches, data.traj_moves):
            optimizer.zero_grad()
            logp = net(state_embeds, actions)
            loss = -logp
            # print(f"loss: {loss}")
            train_loss += loss
            # print(f"train_loss: {train_loss}")
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}\t"
              + f"train loss: {loss}\t"
              + f"({time.time() - start:.0f}s)")
        train_losses.append(train_loss)

    # torch.save(abstract_net.state_dict(), 'abstract_net.pt')


def sample_micro_trajectories(net, data):
    for i in range(len(data.trajs)):
        points = data.points_lists[i]
        (x, y, x_goal, y_goal) = points[0][0]
        moves = ''.join(data.trajs[i])
        print(f'({x, y}) to ({x_goal, y_goal}) via {moves}')
        moves_taken = ''
        for j in range(data.seq_len):
            if max(x, y) == data.max_coord:
                break
            state_embed = data.embed_state((x, y, x_goal, y_goal))
            state_batch = torch.unsqueeze(state_embed, 0)
            action_logits, _, _ = net.controller(state_batch)
            action = torch.argmax(action_logits[:, 0, :])
            # print(f"action: {action}")
            x, y = TrajData.execute((x, y), action)
            move = 'R' if action == 0 else 'U'
            moves_taken += move
            # print(f'now at ({x, y})')
        print(f'({0, 0}) to ({x, y}) via {moves_taken}')
        print('-'*10)


def sample_trajectories(net, data):
    """
    To sample with options:
    1. sample option from start state.
    2. choose actions according to option policy until stop.
    3. after stopping, sample new option.
    4. repeat until done.
    """
    if net.model == 'micro':
        sample_micro_trajectories(net, data)
        return

    for i in range(len(data.trajs)):
        points = data.points_lists[i]
        (x, y, x_goal, y_goal) = points[0][0]
        moves = ''.join(data.trajs[i])
        print(f'({x, y}) to ({x_goal, y_goal}) via {moves}')
        moves_taken = ''
        options = []
        current_option = ''
        option = None
        for j in range(data.seq_len):
            if max(x, y) == data.max_coord:
                break
            state_embed = data.embed_state((x, y, x_goal, y_goal))
            state_batch = torch.unsqueeze(state_embed, 0)
            action_logits, stop_logits, start_logits = net.controller(state_batch)
            if option is None:
                option = Categorical(logits=start_logits).sample()
            else:
                # possibly stop previous option!
                stop = Categorical(logits=stop_logits[0, option, :]).sample()
                if stop:  # zero means keep going!
                    option = Categorical(logits=start_logits[0]).sample()
                    options.append(current_option)
                    current_option = ''

            current_option += str(option.item())
            action = Categorical(logits=action_logits[0, option, :]).sample()
            # print(f"action: {action}")
            x, y = TrajData.execute((x, y), action)
            move = 'R' if action == 0 else 'U'
            moves_taken += move
            # print(f'now at ({x, y})')
        options.append(current_option)
        print(f'({0, 0}) to ({x, y}) via {moves_taken}')
        print(f"options: {'/'.join(options)}")
        print('-'*10)


def logsumexp(tensor, dim=-1, mask=None):
    """taken from https://github.com/pytorch/pytorch/issues/32097"""
    if mask is None:
        mask = torch.ones_like(tensor)
    else:
        assert mask.shape == tensor.shape, 'The factors tensor should have the same shape as the original'
    a = torch.cat([torch.max(tensor, dim, keepdim=True) for _ in range(tensor.shape[dim])], dim)
    return a + torch.sum((tensor - a).exp()*mask, dim).log()


def logaddexp(tensor, other, mask=None):
    if mask is None:
        mask = torch.tensor([1, 1])
    else:
        assert mask.shape == (2, ), 'invalid mask provided'

    a = torch.max(tensor, other)
    # clamp to get rid of nans
    # https://github.com/pytorch/pytorch/issues/1620
    # calculation from https://en.wikipedia.org/wiki/Log_probability#Addition_in_log_space
    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1]).clamp(min=1E-8).log()


def logsubexp(tensor, other):
    return logaddexp(tensor, other, mask=torch.tensor([1, -1]))


def log_practice():
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.1, 0.9])

    c = torch.tensor([0.75, 0.25])
    a2, b2, c2 = map(torch.log, (a, b, c))

    # print( (a + b).log())
    # print( logaddexp(a2, b2))
    # print( (a - b).log())
    # print( logaddexp(a2, b2, torch.tensor([1., -1.])))

    macro_stops = a * b
    print(f"macro_stops: {macro_stops}")
    still_there = a - macro_stops
    print(f"still_there: {still_there}")
    total_new = sum(macro_stops)
    print(f"total_new: {total_new}")
    redist = total_new * c
    print(f"redist: {redist}")
    new_macro = still_there + redist
    print(f"new_macro: {new_macro}")

    assert torch.isclose(torch.logsumexp(a2, dim=0), torch.tensor(0.))
    macro_stops2 = a2 + b2
    print(f"macro_stops2: {macro_stops2}")
    print(torch.isclose(macro_stops2, torch.log(macro_stops)))
    still_there2 = logsubexp(a2, macro_stops2)
    print(f"still_there2: {still_there2}")
    print(torch.isclose(still_there2, torch.log(still_there)))
    total_new2 = torch.logsumexp(still_there2, dim=0)
    print(torch.isclose(total_new2, torch.log(total_new)))
    redist2 = total_new2 + c2
    print(torch.isclose(redist2, torch.log(redist)))
    new_macro2 = torch.logaddexp(still_there2, redist2)
    print(torch.isclose(new_macro2, torch.log(new_macro)))
    print(torch.logsumexp(new_macro2, dim=0))
    print(new_macro2.exp())
    print(torch.isclose(torch.sum(new_macro2.exp()), torch.tensor(1.)))
    print(torch.isclose(torch.logsumexp(new_macro2, dim=0), torch.tensor(0.)))


def main():
    random.seed(1)
    torch.manual_seed(1)

    scale = 3
    seq_len = 5
    trajs = generate_data(scale, seq_len, n=100)

    data = TrajData(trajs)
    print(f"Number of trajectories: {len(data.traj_batches)}")

    n_abstractions = 2
    model = 'HMM'
    # model = 'DP'
    # model = 'micro'

    net = Eq2Net(n_abstractions, data.state_dim, n_micro_actions=2,
                 abstract_penalty=0, model=model)
    # utils.load_model(net, f'models/model_9-10_{model}.pt')
    train_abstractions(data, net, epochs=15)
    utils.save_model(net, f'models/model_9-10_{model}.pt')

    eval_data = TrajData(generate_data(scale, seq_len, n=10),
                         max_coord=data.max_coord)
    # sample_trajectories(net, eval_data)


if __name__ == '__main__':
    # log_practice()
    # torch.autograd.set_detect_anomaly(True)
    main()
