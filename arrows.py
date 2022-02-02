import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset  # , DataLoader
import utils
from utils import assert_equal
from torch.distributions import Categorical


def arrow_path(scale=3, max_macro_repeats=3, seq_len=10):
    path = ''
    while True:
        dir = '<' if random.random() < 0.5 else '>'
        dirs = dir * random.randint(1, max_macro_repeats)
        path += dirs
        if len(path) >= seq_len:
            path = path[:seq_len]
            break
        path += '^'

    path += '*'

    expanded_path = ('.' * (scale - 1)).join(path)
    return expanded_path


def generate_arrow_data(scale=3, seq_len=10, n=100):
    return [arrow_path(scale=scale, max_macro_repeats=2, seq_len=seq_len)
            for _ in range(n)]


class ArrowData(Dataset):
    def __init__(self, trajs, scale):
        """
        path: list of trajectories a la generate_arrow_data()

        generates (obs, move) data points from the trajs, and
        makes a dataset of them.

        0 = up = '^'
        1 = left = '<'
        2 = right = '>'
        """
        super().__init__()
        # strings '<....>....^....'
        self.trajs = trajs
        self.seq_len = len(self.trajs[0])
        self.scale = scale
        # not necessary, but currently assuming in code elsewhere
        assert all(len(t) == self.seq_len for t in self.trajs)

        self.state_dim = 5  # <>^.*

        self.states = [torch.stack([self.embed(p) for p in traj])
                             for traj in self.trajs]
        self.moves = [self.moves_for(t) for t in self.trajs]
        self.traj_coord_dicts = [self.coord_dict(traj) for traj in self.trajs]

    def coord_dict(self, traj):
        d = {}
        (x, y) = (0, 0)
        for point in traj.replace('.', ''):
            d[(x, y)] = self.embed(point)

            if point == '*':
                return d

            for _ in range(self.scale):
                (x, y) = ArrowData.execute((x, y), ArrowData.move(point))

    def get_state(self, i, point):
        if point in self.traj_coord_dicts[i]:
            return self.traj_coord_dicts[i][point]
        else:
            return self.embed('.')

    def moves_for(self, traj):
        simple_traj = traj.replace('.', '')
        # ignore * at end
        simple_moves = [ArrowData.move(arrow) for arrow in simple_traj[:-1]]
        moves = [[m] * self.scale for m in simple_moves]
        moves = [m for lst in moves for m in lst]
        return torch.tensor(moves)

    def move(arrow):
        return '^<>'.index(arrow)

    def arrow(move):
        return '^<>'[move]

    def execute(point, move):
        (x, y) = point
        if move == 0:
            return (x, y + 1)
        elif move == 1:
            return (x - 1, y)
        else:
            assert move == 2
            return (x + 1, y)

    def embed(self, point):
        return F.one_hot(torch.tensor('.^<>*'.index(point)),
                         num_classes=self.state_dim).float()

    def make_points(traj):
        trace = ArrowData.exec_traj(traj)
        assert_equal(len(trace), len(traj) + 1)
        goal = trace[-1]

        # list of ((x, y, x_goal, y_goal), move) tuples
        points = [((*point, *goal),
                   torch.tensor(0 if move == 'R' else 1))
                  for point, move in zip(trace[:-1], traj)]
        # needed so we can probe probability of stopping at the end!
        points.append(((*goal, *goal), torch.tensor(-1)))
        return points

    def __len__(self):
        return len(self.traj_moves)

    def __getitem__(self, idx):
        return (self.traj_batches[idx],
                self.traj_moves[idx])


def sample_arrow_trajs(net, data):
    """
    To sample with options:
    1. sample option from start state.
    2. choose actions according to option policy until stop.
    3. after stopping, sample new option.
    4. repeat until done.
    """
    if net.model == 'micro':
        raise NotImplementedError()

    for i in range(len(data.trajs)-1):
        print(f'path1: {data.trajs[i]}')
        moves_taken = ''
        options = []
        current_option = ''
        option = None
        (x, y) = (0, 0)
        for j in range(data.seq_len):
            state_embed = data.get_state(i, (x, y))
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
            (x, y) = ArrowData.execute((x, y), action)
            moves_taken += ArrowData.arrow(action)
            # print(f'now at ({x, y})')
        options.append(current_option)
        print(f'path2: {moves_taken}')
        print(f"options: {'/'.join(options)}")
        print('-'*10)

# def main():
#     random.seed(1)
#     torch.manual_seed(1)

#     scale = 5
#     seq_len = 3
#     trajs = generate_arrow_data(scale=scale, seq_len=seq_len, n=200)

#     data = ArrowData(trajs, scale)
#     print(f"Number of trajectories: {len(data.traj_batches)}")

#     n_abstractions = 3
#     model = 'HMM'
#     # model = 'DP'
#     # model = 'micro'

#     net = abstraction3.Eq2Net(n_abstractions, data.state_dim, n_micro_actions=3,
#                               abstract_penalty=0.0, model=model)
#     # utils.load_model(net, f'models/model_9-10_{model}.pt')
#     abstraction3.train_abstractions(data, net, epochs=50, lr=1E-3)
#     utils.save_model(net, f'models/arrow_9-10_{model}.pt')

#     eval_data = ArrowData(generate_arrow_data(scale=scale, seq_len=seq_len, n=10),
#                           scale=scale)
#     sample_arrow_trajs(net, eval_data)


if __name__ == '__main__':
    # main()
    scale = 5
    data = ArrowData(generate_arrow_data(n=10, seq_len=5, scale=scale), scale=scale)
    for i in range(10):
        print(data.trajs[i])
        # print(data.traj_moves[i])
        # print(data.traj_batches[i])
        # print(data.traj_coord_dicts[i])
