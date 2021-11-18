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
        self.stop_net_stop_ix = 0  # index zero means stop, index 1 means go
        self.stop_net_continue_ix = 1  # index zero means stop, index 1 means go

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
           (T, b, t, t) tensor of causal consistency penalties
        """
        T = s_i.shape[0]

        t_i = self.tau_net(s_i)  # (T, t)
        action_logps = self.micro_net(s_i).reshape(T, self.b, self.a)
        stop_logps = self.stop_net(s_i).reshape(T, self.b, 2)
        start_logps = self.start_net(t_i)  # (T, b)
        consistency_penalty = self.calc_consistency_penalty(t_i)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return t_i, action_logps, stop_logps, start_logps, consistency_penalty

    def calc_consistency_penalty(self, t_i):
        T = t_i.shape[0]
        # calculate transition for each t_i + b pair
        t_i2 = t_i.repeat_interleave(self.b, dim=0)  # (T*b, t)
        assertEqual(t_i2.shape, (T*self.b, self.t))
        b_onehots = F.one_hot(torch.arange(self.b)).repeat(T, 1)  # (T*b, b)
        assertEqual(b_onehots.shape, (T*self.b, self.b))
        # b is "less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*b, t + b)
        assertEqual(t_i2.shape, (T*self.b, self.t + self.b))
        # (T * b, t + b) -> (T * b, t)
        t_i2 = self.alpha_net(t_i2)
        t_i2 = t_i2.reshape(T, 1, self.b, self.t)
        t_i3 = t_i.reshape(1, T, 1, self.t)
        penalty = (t_i2 - t_i3)**2
        assertEqual(penalty.shape, (T, T, self.b, self.t))
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty


class Eq2Net(nn.Module):
    def __init__(self, abstract_policy_net, abstract_penalty=0.5,
                 consistency_ratio=1.):
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
        self.consistency_ratio = consistency_ratio

    def forward(self, s_i, actions):
        """
        s_i: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: logp of sequence

        HMM calculation, building off Smith et al. 2018.
        """
        T = len(actions)
        assertEqual(s_i.shape, (T + 1, self.s))
        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        t_i, action_logps, stop_logps, start_logps, consistency_penalties = self.abstract_policy_net(s_i)

        total_logp = 0.
        total_consistency_penalty = 0.
        # (i+1, b) dist over options keeps track of when option started.
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
                stop_lps = stop_logps[i, :, self.abstract_policy_net.stop_net_stop_ix]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, self.abstract_policy_net.stop_net_continue_ix]  # (b,)
                start_lps = start_logps[i]  # (b,)

                # prob mass for options exiting which started at step i; broadcast
                option_step_stops = option_step_dist + stop_lps.reshape(1, self.b)  # (i+1, b)
                total_rearrange = torch.logsumexp(option_step_stops, dim=(0, 1))
                total_rearrange = total_rearrange - self.abstract_penalty
                # distribute new mass among new options. broadcast
                new_mass = start_lps + total_rearrange  # (b,)

                # mass that stays in place, aka doesn't stop; broadcast
                option_step_dist = option_step_dist + one_minus_stop_lps  # (T, b)
                # add new mass
                option_step_dist = torch.cat((option_step_dist,
                                             new_mass.unsqueeze(0)))

                # causal consistency penalty
                consistency_pens = consistency_penalties[:i+1, i, :]  # (i+1, b)
                assertEqual(consistency_pens.shape, (i+1, self.b))
                consistency_penalty = torch.logsumexp(option_step_dist + consistency_pens, dim=(0, 1))
                total_consistency_penalty += consistency_penalty

            action_lps = action_logps[i, :, action]  # (b,)
            # in prob space, this is a sum of probs weighted by macro-dist
            logp = torch.logsumexp(action_lps + option_step_dist, dim=(0, 1))
            total_logp += logp

        # all macro options need to stop at the very end.
        final_stop_lps = stop_logps[-1, :, self.abstract_policy_net.stop_net_stop_ix]  # (b,)
        # broadcast
        total_logp += torch.logsumexp(final_stop_lps.reshape(1, self.b) + option_step_dist, dim=(0, 1))

        # maximize logp, minimize causal inconsistency
        loss = -total_logp + self.consistency_ratio * total_consistency_penalty
        return loss


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
            loss = net(s_i, actions)
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


# def sample_trajectories2(net, data):
#     """
#     1. tau(s_0) = t_0
#     2. sample b given t_0.
#     3. execute b until you stop, generating some s_i, a_i stuff.
#     4. run alpha(t_0, b) = t_1.
#     5. repeat steps 2–4 until you generate enough s_i, a_i to get to end.
#     """
#     for i in range(len(data.trajs)):
#         points = data.points_lists[i]
#         (x, y, x_goal, y_goal) = points[0][0]
#         moves = ''.join(data.trajs[i])
#         print(f'({x, y}) to ({x_goal, y_goal}) via {moves}')
#         moves_taken = ''
#         options = []
#         current_option = ''
#         option = None

#         for j in range(data.seq_len):
#             if max(x, y) == data.max_coord:
#                 break
#             state_embed = data.embed_state((x, y, x_goal, y_goal))
#             state_batch = torch.unsqueeze(state_embed, 0)
#             # only use action_logps, stop_logps, and start_logps
#             t_i, action_logps, stop_logps, start_logps, causal_penalty = net.abstract_policy_net(state_batch)
#             if option is None:
#                 option = Categorical(logits=start_logps).sample()
#             else:
#                 # possibly stop previous option!
#                 stop = Categorical(logits=stop_logps[0, option, :]).sample()
#                 if stop == net.abstract_policy_net.stop_net_stop_ix:
#                     option = Categorical(logits=start_logps[0]).sample()
#                     options.append(current_option)
#                     current_option = ''

#             current_option += str(option.item())
#             action = Categorical(logits=action_logps[0, option, :]).sample()
#             # print(f"action: {action}")
#             x, y = up_right.TrajData.execute((x, y), action)
#             move = 'R' if action == 0 else 'U'
#             moves_taken += move
#             # print(f'now at ({x, y})')

#         options.append(current_option)
#         print(f'({0, 0}) to ({x, y}) via {moves_taken}')
#         print(f"options: {'/'.join(options)}")
#         print('-'*10)


def sample_trajectories(net, data):
    """
    To sample with options:
    1. sample option from start state.
    2. choose actions according to option policy until stop.
    3. after stopping, sample new option.
    4. repeat until done.
    """
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
            # only use action_logps, stop_logps, and start_logps
            t_i, action_logps, stop_logps, start_logps, causal_penalty = net.abstract_policy_net(state_batch)
            if option is None:
                option = Categorical(logits=start_logps).sample()
            else:
                # possibly stop previous option!
                stop = Categorical(logits=stop_logps[0, option, :]).sample()
                if stop == net.abstract_policy_net.stop_net_stop_ix:
                    option = Categorical(logits=start_logps[0]).sample()
                    options.append(current_option)
                    current_option = ''

            current_option += str(option.item())
            action = Categorical(logits=action_logps[0, option, :]).sample()
            # print(f"action: {action}")
            x, y = up_right.TrajData.execute((x, y), action)
            move = 'R' if action == 0 else 'U'
            moves_taken += move
            # print(f'now at ({x, y})')
        options.append(current_option)
        print(f'({0, 0}) to ({x, y}) via {moves_taken}')
        print(f"options: {'/'.join(options)}")
        print('-'*10)


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
                 abstract_penalty=.00185,
                 consistency_ratio=1.0)

    # utils.load_model(net, f'models/model_9-10_{model}.pt')
    train_abstractions(data, net, epochs=25)
    # utils.save_model(net, f'models/model_9-17.pt')

    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)
    sample_trajectories(net, eval_data)


if __name__ == '__main__':
    # log_practice()
    # torch.autograd.set_detect_anomaly(True)
    main()
