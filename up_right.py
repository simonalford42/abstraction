import torch
from torch.utils.data import Dataset  # , DataLoader
import random
from utils import assertEqual
import torch.nn.functional as F


def generate_data(scale, seq_len, n):
    return [generate_grid_traj(scale, seq_len) for _ in range(n)]


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
