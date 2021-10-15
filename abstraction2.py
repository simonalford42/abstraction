import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import assertEqual, num_params, FC


class AbstractController(nn.Module):
    """
    Input: [abstract_action_embed | state_embed]
    Output: [micro_action_logits | stop_prob]
    """
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        self.n_abstractions = n_abstractions
        self.state_dim = state_dim
        self.n_micro_actions = n_micro_actions
        self.net = FC(input_dim=n_abstractions + state_dim,
                      output_dim=n_micro_actions + 1,
                      num_hidden=2,
                      hidden_dim=16)

    def forward(self, x):
        """
        x: batch of vectors which are [abstraction_embed | state_embed]
        returns: batch of (micro_action_logits, stop_prob)
        """
        B = len(x)
        out = self.net(x)
        action_logits = out[:, :-1]
        assertEqual(action_logits.shape, (B, self.n_micro_actions))
        stop_probs = out[:, -1]
        assertEqual(stop_probs.shape, (B, ))
        return action_logits, stop_probs

    def prob_batched(self, abstract_actions, states):
        """
        abstract_actions: (B, ) ints
        states: (B, state_dim)
        """
        B = len(abstract_actions)
        abstract_embeds = F.one_hot(abstract_actions,
                                    num_classes=self.n_abstractions)
        assertEqual(abstract_embeds.shape, (B, self.n_abstractions))
        assertEqual(states.shape, (B, self.state_dim))
        state_action_embed = torch.cat([abstract_embeds, states], dim=1)
        assertEqual(state_action_embed.shape,
                    (B, self.n_abstractions + self.state_dim))
        return self(state_action_embed)

    def prob(self, abstract_action, state):
        """
        abstract_action: int
        state: vec
        returns: (micro_action_logits, stop_prob)

        This batches and unbatches for you.
        """
        abstract_embed = F.one_hot(torch.tensor(abstract_action),
                                   num_classes=self.n_abstractions)
        # batch of one
        state_action_embed = torch.cat([abstract_embed, state]).unsqueeze(0)
        action_logits, stop_probs = self(state_action_embed)
        assertEqual(action_logits.shape, (1, self.n_micro_actions))
        assertEqual(stop_probs.shape, (1, ))
        # unbatch
        return action_logits[0], stop_probs[0]


class Eq1Net(nn.Module):
    """
    Input: micro traj, initial state, abstract action
    Output: probability of traj given initial state, abstract action
    """
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        self.n_abstractions = n_abstractions
        self.state_dim = state_dim
        self.n_micro_actions = n_micro_actions
        self.controller = AbstractController(n_abstractions,
                                             state_dim,
                                             n_micro_actions)

    # def forward(self, states, actions, abstract_action):
    #     return self.forward_batched(states, actions, abstract_action)
    def forward(self, traj, abstract_action):
        return self.forward_unbatched(traj, abstract_action)

    def forward_unbatched(self, traj, abstract_action):
        """
        traj: vector of (state, action) tuples.
        (last state's action isn't used, could be -1 marking end of seq.)
        state is vector of dim state_dim
        action is an int
        abstract_action: int of abstract action

        returns: P(state/action sequence | abstract action)
        """
        p = torch.tensor(1.)
        for state, action in traj[:-1]:
            action_logits, prob_stop = self.controller.prob(abstract_action, state)
            prob_action = action_logits[action]
            p = p * prob_action * (1 - prob_stop)

        final_state = traj[-1][0]
        _, prob_stop = self.controller.prob(abstract_action, final_state)
        p *= prob_stop

        return p

    def forward_batched(self, states, actions, abstract_action):
        """
        states: (B, state_dim, n + 1) tensor of state sequences
        actions: (B, n) tensor of actions
        abstract_action: int
        """
        (B, n) = actions.shape
        assertEqual(states.shape, (B, self.state_dim, n + 1))
        abstract_actions = torch.full((B, ), abstract_action)

        trans_states = states.reshape(n + 1, B, self.state_dim)
        trans_actions = actions.reshape(n, B)
        p = torch.ones(B)
        # each iter is a batch of state/action at step i
        # ignore final state
        for states, actions in zip(trans_states[:-1], trans_actions):
            assertEqual(states.shape, (B, self.state_dim))
            assertEqual(actions.shape, (B, ))
            action_logits, prob_stop = self.controller.prob_batched(
                abstract_actions, states)
            # get prob of chosen action for each item in batch
            # see https://numpy.org/doc/stable/reference/arrays.indexing.html#purely-integer-array-indexing
            prob_action = action_logits[range(B), actions]
            assertEqual(prob_action.shape, (B, ))
            assertEqual(prob_stop.shape, (B, ))
            p = p * prob_action * (1 - prob_stop)

        final_states = trans_states[-1]
        _, prob_stop = self.controller.prob_batched(abstract_actions,
                                                    final_states)
        p *= prob_stop

        return p


class Eq2Net(nn.Module):
    """
    Input: traj
    Output: probability of traj, which can be trained via backprop
    Does so via DP
    """
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        self.n_abstractions = n_abstractions
        self.state_dim = state_dim
        self.n_micro_actions = n_micro_actions
        self.eq1_net = Eq1Net(n_abstractions,
                              state_dim,
                              n_micro_actions)

    def f(self, traj):
        return sum(self.eq1_net(traj, i)
                   for i in range(self.n_abstractions))

    def f_batched(self, states, actions):
        """
        states: (B, state_dim, n + 1)
        actions: (B, n)
        """
        # we could super-batch this by parallelizing over abstract action
        # could check directly with unbatched as it gets summed
        return sum(self.eq1_net(states, actions, i)
                   for i in range(self.n_abstractions))

    # def forward(self, states, actions):
    #     return self.forward_batched(states, actions)
    def forward(self, traj):
        return self.forward_unbatched(traj)

    def forward_unbatched(self, traj):
        """
        traj: list of (s_i, a_i) where a[-1] is -1
        s_i is a vector, a_i is an int
        """
        assert traj[-1][1] == -1

        # tabulated results
        p_table = [0] * (len(traj) + 1)
        p_table[-1] = 1

        for a in range(len(traj) - 1, -1, -1):
            p = sum(self.f(traj[a:b + 1]) * p_table[b + 1]
                    for b in range(a, len(traj)))
            p /= self.n_abstractions
            p_table[a] = p

        # assert p_table[-2] == self.f(traj[-1:]) / self.n_abstractions

        return p_table[0]

    def forward_batched(self, states, actions):
        """
        states: (B, state_dim, n + 1)
        actions: (B, n)
        returns: (B, ) probabilities
        """
        B, n = actions.shape
        assertEqual(states.shape, (B, self.state_dim, n + 1))

        p_table = [torch.zeros(B) for _ in range(n + 2)]
        p_table[-1] = torch.ones(B)

        for a in range(n, -1, -1):
            p = sum(self.f_batched(states[:, :, a:b + 1], actions[:, a:b])
                    * p_table[b + 1]
                    for b in range(a, n + 1))
            assertEqual(p.shape, (B, ))
            p /= self.n_abstractions
            p_table[a] = p

        return p_table[0]

    def forward_debug(self, traj):
        # check DP calculation on a simpler function
        def f2(lst):
            return sum(lst)

        # tabulated results
        p_table = [0] * (len(traj) + 1)
        p_table[-1] = 1

        for a in range(len(traj) - 1, -1, -1):
            p = sum(f2(traj[a:b + 1]) * p_table[b + 1]
                    for b in range(a, len(traj)))
            p /= 1
            p_table[a] = p

        print(p_table)


class PolicyNet(nn.Module):
    """
    Input: current state as a tuple (x, y, x_goal, y_goal)
    Output: probability of acting U/D/L/R.
    """
    def __init__(self, max_coord):
        super().__init__()
        # max x or y coordinate
        self.max_coord = max_coord
        self.state_dim = 4 * (self.max_coord + 1)
        self.fc1 = nn.Linear(self.state_dim, 2)

    def forward(self, x):
        """
        x: batch of (B, 4 * (max_coord + 1))
        out: batch of (up, right) logits
        """
        assertEqual(x.shape[1], 4 * (self.max_coord + 1))
        x = self.fc1(x)
        return x


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

        # list of list of ((x, y, x_goal, y_goal), move) tuples
        self.points_lists = [TrajData.make_points(t) for t in trajs]
        # list of ((x, y, x_goal, y_goal), move) tuples
        self.points = [p for points in self.points_lists
                       for p in points if p[1] != -1]

        # list of (x, y, x_goal, y_goal)
        # list of moves (int)
        self.coords, self.moves = zip(*self.points)
        self.max_coord = max(max(c) for c in self.coords)
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

        # each trajectory as a batch of state_embeds
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


class TrajData2(Dataset):
    def __init__(self, traj_data: TrajData, probs):
        super().__init__()
        self.traj_data = traj_data
        self.probs = probs

    def __len__(self):
        return len(self.traj_data.trajs)

    def __getitem__(self, idx):
        """
        (state_embeds: [state_dim, n+1],
         actions: [n]), prob
        """
        return ((self.traj_data.traj_batches[idx].transpose(0, 1),
                 self.traj_data.traj_moves[idx]),
                self.probs[idx])


def train_policy_net(data, net, epochs):
    batch_size = 32

    print(f"net has {num_params(net)} parameters")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_errors = []
    net.train()
    for epoch in range(epochs):
        train_loss = 0

        for examples, labels in dataloader:
            optimizer.zero_grad()
            # examples: [batch_size, (x, y, x_goal, y_goal)]
            # labels: [batch_size]
            logits = net(examples)
            loss = criterion(logits, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_error = model_error(dataloader, net)

        print(f"epoch: {epoch}\t"
              + f"train loss: {loss}\t"
              + f"train error: {train_error}\t")
        train_losses.append(train_loss)
        train_errors.append(train_error)


def model_error(dataloader, model):
    total_wrong = 0
    n = 0

    with torch.no_grad():
        model.eval()

        for examples, labels in dataloader:
            n += len(labels)
            logits = model(examples)
            preds = torch.argmax(logits, dim=1)
            num_wrong = (preds != labels).sum()
            total_wrong += num_wrong

    return total_wrong / n


def eval_policy_net(data, n_trajs, net):
    net.eval()
    with torch.no_grad():
        for i in range(n_trajs):
            traj = data.trajs[i]
            print(f"traj: {traj}")
            batch = data.traj_batches[i]
            probs = F.softmax(net(batch), dim=1)
            probs = [round(p[0].item(), 2) for p in probs]
            print(probs)


def traj_prob(batch, actions, policy_net):
    """
    state_embeds: list of state_embeds
    out: probability of each step
    """
    policy_net.eval()
    with torch.no_grad():
        probs = F.softmax(policy_net(batch), dim=1)
        probs = torch.tensor([p[a] for a, p in zip(actions, probs)])
        return torch.prod(probs)

    return torch.prod(probs)


def train_abstractions(data: TrajData, abstract_net, target_policy_net, epochs):
    print(f"abstract net has {num_params(abstract_net)} parameters")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(abstract_net.parameters(), lr=0.001)

    train_losses = []
    train_errors = []
    abstract_net.train()

    traj_probs = [traj_prob(b, m, target_policy_net)
                  for b, m in zip(data.traj_batches, data.traj_moves)]

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for traj_embed, target_p in zip(data.traj_embeds, traj_probs):
            optimizer.zero_grad()
            p = abstract_net(traj_embed)
            loss = criterion(p, target_p)
            print(f"loss: {loss}")
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_error = 0  # TODO

        print(f"epoch: {epoch}\t"
              + f"train loss: {loss}\t"
              # + f"train error: {train_error}\t"
              + f"({time.time() - start:.0f}s)")
        train_losses.append(train_loss)
        train_errors.append(train_error)


def train_abstractions_batched(
        data: TrajData, abstract_net,
        target_policy_net, epochs):
    print(f"abstract net has {num_params(abstract_net)} parameters")
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(abstract_net.parameters(), lr=0.001)

    train_losses = []
    train_errors = []
    abstract_net.train()

    traj_probs = [traj_prob(b, m, target_policy_net)
                  for b, m in zip(data.traj_batches, data.traj_moves)]
    data2 = TrajData2(data, traj_probs)
    dataloader = DataLoader(data2, batch_size=1, shuffle=False)

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for (states, actions), target_probs in dataloader:
            optimizer.zero_grad()
            probs = abstract_net(states, actions)
            loss = criterion(probs, target_probs)
            print(f"loss: {loss}")
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_error = 0  # TODO

        print(f"epoch: {epoch}\t"
              + f"train loss: {loss}\t"
              # + f"train error: {train_error}\t"
              + f"({time.time() - start:.0f}s)")
        train_losses.append(train_loss)
        train_errors.append(train_error)


def eval_abstractions(data, n_trajs, abstract_net, n_abstractions):
    abstract_net.eval()
    with torch.no_grad():
        for i in range(n_trajs):
            traj = data.trajs[i]
            traj_embed = data.traj_embeds[i]
            print('\t' + ''.join(traj))
            for abstract_action in range(n_abstractions):
                # for start in range(len(traj_embed)):
                for start in range(4):
                    # for end in range(start + 1, len(traj_embed)):
                    for end in range(start + 1, start + 4):
                        prob = abstract_net.eq1_net(traj_embed[start:end + 1],
                                                    abstract_action)
                        # states_embed = torch.stack(
                        #     data.traj_states[i][start:end + 1]).transpose(0, 1)
                        # assertEqual(states_embed.shape,
                        #             (data.state_dim, end - start + 1))
                        # actions = data.traj_moves[i][start:end]
                        # assertEqual(actions.shape, (end - start, ))
                        # prob = abstract_net.eq1_net(
                        #     states_embed.unsqueeze(0),
                        #     actions.unsqueeze(0),
                        #     abstract_action)
                        # assertEqual(prob.shape, (1,))
                        # prob = prob[0]

                        print(f"{abstract_action} {prob:.2f}\t"
                              + ('-' * start)
                              + traj[start:end]
                              + ('-' * max(0, len(traj) - end)))


def test_batched_eq_nets():
    random.seed(1)
    torch.manual_seed(1)

    scale = 3
    seq_len = 5
    trajs = generate_data(scale, seq_len, n=5)
    data = TrajData(trajs)
    policy_net = PolicyNet(max_coord=data.max_coord)
    train_policy_net(data, policy_net, epochs=10)
    eval_policy_net(data, n_trajs=5, net=policy_net)

    n_abstractions = 2
    state_dim = policy_net.state_dim
    abstract_net = Eq2Net(n_abstractions, state_dim, n_micro_actions=2)
    # abstract_net.forward_debug([1, 2, 3, 4, 5])

    train_abstractions(data, abstract_net, policy_net, epochs=1)
    # train_abstractions_batched(data, abstract_net, policy_net, epochs=1)
    eval_abstractions(data, n_trajs=1, abstract_net=abstract_net,
                      n_abstractions=2)


def main():
    random.seed(1)
    torch.manual_seed(1)

    scale = 3
    seq_len = 5
    trajs = generate_data(scale, seq_len, n=100)
    data = TrajData(trajs)
    policy_net = PolicyNet(max_coord=data.max_coord)
    train_policy_net(data, policy_net, epochs=100)
    eval_policy_net(data, n_trajs=5, net=policy_net)

    n_abstractions = 2
    state_dim = policy_net.state_dim
    abstract_net = Eq2Net(n_abstractions, state_dim, n_micro_actions=2)
    # abstract_net.forward_debug([1, 2, 3, 4, 5])

    train_abstractions(data, abstract_net, policy_net, epochs=20)
    eval_abstractions(data, n_trajs=5, abstract_net=abstract_net,
                      n_abstractions=2)


if __name__ == '__main__':
    test_batched_eq_nets()
