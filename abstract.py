import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset  # , DataLoader
# import utils
from utils import assertEqual, num_params, FC
from torch.distributions import Categorical
import up_right


class AbstractPolicyNet(nn.Module):
    def __init__(self, a, b, s, t, tau_net, micro_net, stop_net, start_net,
                 alpha_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        self.s = s  # state dim
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b, a)  = P(a | b, s)
        self.stop_net = stop_net  # s -> 2b of (stop, 1 - stop) aka beta.
        self.start_net = start_net  # t -> b  aka P(b | t)
        self.alpha_net = alpha_net  # (t + b) -> t abstract transition

    def forward(self, s_i):
        """
        input: (T, s) tensor of states
        outputs:
           (T, t) tensor of abstract states
           (T, b, n) tensor of action logps
           (T, b, 2) tensor of stop logps
               interpretation: (T, b, 0) is "keep going" (stop=False) (1 - beta)
                               (T, b, 1) is stop logp
           (T, b) tensor of start logps
        """
        T = s_i.shape[0]

        t_i = self.tau_net(s_i)  # (T, t)
        action_logps = self.micro_net(s_i).reshape(T, self.b, self.a)
        stop_logps = self.stop_net(s_i).reshape(T, self.b, 2)
        start_logps = self.start_net(t_i)  # (T, b)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return t_i, action_logps, stop_logps, start_logps


class Eq2Net(nn.Module):
    def __init__(self, abstract_policy_net, abstract_penalty=0.5, model='HMM'):
        """
        model:
            DP: dynamic programming model
            HMM: HMM model
            micro: no options.
        """
        super().__init__()
        self.abstract_policy_net = abstract_policy_net
        self.a = abstract_policy_net.a
        self.b = abstract_policy_net.b
        self.s = abstract_policy_net.s
        self.t = abstract_policy_net.t

        # logp penalty for longer sequences
        self.abstract_penalty = abstract_penalty
        assert model in ['DP', 'HMM', 'micro']
        self.model = model

    def forward(self, s_i, actions):
        if self.model == 'DP':
            return self.forward_DP(s_i, actions)
        elif self.model == 'HMM':
            return self.forward_HMM(s_i, actions)
        elif self.model == 'micro':
            return self.forward_micro(s_i, actions)
        else:
            assert False

    def forward_HMM(self, s_i, actions):
        """
        s_i: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: logp of sequence

        HMM calculation, identical to Smith et al. 2018.
        """
        T = len(actions)
        assertEqual(s_i.shape, (T + 1, self.s))
        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b)
        t_i, action_logps, stop_logps, start_logps = self.abstract_policy_net(s_i)

        total_logp = 0.
        # (n_steps_so_far, b) dist over options keeps track of when option started.
        option_step_dist = start_logps[0].unsqueeze(0)
        for i, action in enumerate(actions):
            # invariant: prob dist should sum to 1
            # I was getting error of ~1E-7 which got triggered by default value
            # only applies if no abstract penalty
            if not self.abstract_penalty:
                assert torch.isclose((s := torch.logsumexp(option_step_dist, (0, 1))),
                                     torch.tensor(0.), atol=1E-5), \
                       f'Not quite zero: {s}'

            # transition before acting. this way the state at which an option
            # starts is where its first move happens
            # => skip transition for the first step
            if i > 0:
                stop_lps = stop_logps[i, :, 1]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, 0]  # (b,)
                start_lps = start_logps[i]  # (b,)

                # prob mass for options exiting which started at step i; broadcast
                option_step_stops = option_step_dist + stop_lps  # (T, b)
                total_rearrange = torch.logsumexp(option_step_stops, dim=(0, 1))
                total_rearrange = total_rearrange - self.abstract_penalty
                # distribute new mass among new options. broadcast
                new_mass = start_lps + total_rearrange  # (b,)

                # mass that stays in place, aka doesn't stop; broadcast
                option_step_dist = option_step_dist + one_minus_stop_lps  # (T, b)
                # add new mass
                option_step_dist = torch.cat((option_step_dist,
                                             new_mass.unsqueeze(0)))

            action_lps = action_logps[i, :, action]  # (b,)
            # in prob space, this is a sum of probs weighted by macro-dist
            logp = torch.logsumexp(action_lps + option_step_dist, dim=(0, 1))
            total_logp += logp

        # all macro options need to stop at the very end.
        final_stop_lps = stop_logps[-1, :, 0]
        total_logp += torch.logsumexp(final_stop_lps + option_step_dist, dim=(0, 1))

        return total_logp

    def forward_micro(self, s_i, actions):
        """
        s_i: (T+1, state_dim) tensor
        actions: (T, ) tensor of ints

        outputs: logp of sequence
        """
        T = len(actions)
        assert s_i.shape == (T + 1, self.s)
        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.controller(s_i)

        step_logps = torch.stack([action_logps[i, 0, actions[i]]
                                 for i in range(len(actions))])
        return step_logps.sum(axis=0)


def train_abstractions(data, net, epochs, lr=1E-3):
    print(f"net has {num_params(net)} parameters")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = []
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for s_i, actions in zip(data.traj_batches, data.traj_moves):
            optimizer.zero_grad()
            logp = net(s_i, actions)
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
            action_logps, _, _ = net.controller(state_batch)
            action = torch.argmax(action_logps[:, 0, :])
            # print(f"action: {action}")
            x, y = up_right.TrajData.execute((x, y), action)
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
            action_logps, stop_logps, start_logps = net.controller(state_batch)
            if option is None:
                option = Categorical(logps=start_logps).sample()
            else:
                # possibly stop previous option!
                stop = Categorical(logps=stop_logps[0, option, :]).sample()
                if stop:  # zero means keep going!
                    option = Categorical(logps=start_logps[0]).sample()
                    options.append(current_option)
                    current_option = ''

            current_option += str(option.item())
            action = Categorical(logps=action_logps[0, option, :]).sample()
            # print(f"action: {action}")
            x, y = up_right.TrajData.execute((x, y), action)
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
    trajs = up_right.generate_data(scale, seq_len, n=100)

    data = up_right.TrajData(trajs)
    print(f"Number of trajectories: {len(data.traj_batches)}")

    abstract_policy_net = AbstractPolicyNet(
        a := 2, b := 2, s := data.state_dim, t := 10,
        tau_net=FC(s, t, hidden_dim=64, num_hidden=1),
        micro_net=FC(s, b*a, hidden_dim=64, num_hidden=1),
        stop_net=FC(s, b*2, hidden_dim=64, num_hidden=1),
        start_net=FC(t, b, hidden_dim=64, num_hidden=1),
        alpha_net=FC(t + b, t, hidden_dim=64, num_hidden=1))
    net = Eq2Net(abstract_policy_net,
                 abstract_penalty=0)

    # utils.load_model(net, f'models/model_9-10_{model}.pt')
    train_abstractions(data, net, epochs=150)
    # utils.save_model(net, f'models/model_9-17.pt')

    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)
    sample_trajectories(net, eval_data)


if __name__ == '__main__':
    # log_practice()
    # torch.autograd.set_detect_anomaly(True)
    main()
