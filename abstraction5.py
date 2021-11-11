import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset  # , DataLoader
from utils import assertEqual, num_params, FC


class AbstractController(nn.Module):
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        # self.n_abstractions = n_abstractions
        # self.state_dim = state_dim
        # self.n_micro_actions = n_micro_actions
        # self.net = FC(input_dim=state_dim,
        #               output_dim=n_micro_actions,
        #               num_hidden=1,
        #               hidden_dim=64)
        super().__init__()
        self.b = n_abstractions
        self.s = state_dim
        self.n = n_micro_actions
        self.net = FC(input_dim=self.s,
                      # input: (T, s) tensor of states
                      # outputs:
                      #    (T, b, n) tensor of logps
                      #    (T, b) tensor of stop probs
                      #    (T, b) tensor of prob start
                      # so output (T, num_abstractions * (n + 2))
                      output_dim=self.b * (self.n + 2),
                      num_hidden=1,
                      hidden_dim=64)

    def forward(self, state_embeds):
        """
        input: (T, state_dim) tensor of states
        outputs:
           (T, n_micro_actions) tensor of logps
        """
        T = state_embeds.shape[0]
        assertEqual(state_embeds.shape, (T, self.s))
        # (T, num_abstractions * (n + 2))
        out = self.net(state_embeds)
        action_logits = out[:, :self.b * self.n]
        action_logits = action_logits.reshape(-1,
                                              self.b,
                                              self.n)
        action_logits = F.log_softmax(action_logits, dim=2)
        return action_logits[:, 0, :]

class Eq2Net(nn.Module):
    def __init__(self, n_abstractions, state_dim, n_micro_actions):
        super().__init__()
        self.n_abstractions = n_abstractions
        self.state_dim = state_dim
        self.n_micro_actions = n_micro_actions
        self.controller = AbstractController(n_abstractions,
                                             state_dim,
                                             n_micro_actions)

    def forward(self, state_embeds, actions):
        """
        state_embeds: (T+1, state_dim) tensor
        actions: (T, ) tensor of ints

        outputs: P(actions | s_0)
        """
        T = len(actions)
        assert state_embeds.shape == (T + 1, self.state_dim)

        action_logits = self.controller(state_embeds)
        step_logits = torch.stack([action_logits[i,actions[i]] for i in range(len(actions))])
        return step_logits.sum(axis=0)


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


def generate_grid_traj2(scale, n):
    n = n * scale
    traj = ['R'] * int(n/2)
    traj += ['U'] * (n - len(traj))
    return ''.join(traj)


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


def train_abstractions(data: TrajData, net, epochs):
    print(f"net has {num_params(net)} parameters")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    train_losses = []
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for state_embeds, actions in zip(data.traj_batches, data.traj_moves):
            optimizer.zero_grad()
            logp = net(state_embeds, actions)
            loss = -logp
            train_loss += loss
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}\t"
              + f"train loss: {loss}\t"
              + f"({time.time() - start:.0f}s)")
        train_losses.append(train_loss)

    # torch.save(abstract_net.state_dict(), 'abstract_net.pt')


def sample_trajectories(net, data):
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
            action_logits = net.controller(state_batch)
            action = torch.argmax(action_logits)
            # print(f"action: {action}")
            x, y = TrajData.execute((x, y), action)
            move = 'R' if action == 0 else 'U'
            moves_taken += move
            # print(f'now at ({x, y})')
        print(f'({0, 0}) to ({x, y}) via {moves_taken}')
        print('-'*10)


def main():
    random.seed(1)
    torch.manual_seed(1)

    scale = 3
    seq_len = 5
    trajs = generate_data(scale, seq_len, n=100)
    data = TrajData(trajs)
    print(f"Number of trajectories: {len(data.traj_batches)}")

    n_abstractions = 2
    net = Eq2Net(n_abstractions, data.state_dim, n_micro_actions=2)
    train_abstractions(data, net, epochs=100)
    eval_data = TrajData(generate_data(scale, seq_len, n=10))
    sample_trajectories(net, eval_data)


if __name__ == '__main__':
    main()
