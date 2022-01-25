import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import utils
from utils import assertEqual, Timing, DEVICE
from modules import FC, AllConv, RelationalDRLNet
from torch.distributions import Categorical
import old_box_world
import box_world
import up_right
import mlflow


STOP_NET_STOP_IX = 0
STOP_NET_CONTINUE_IX = 1


class AbstractPolicyNet(nn.Module):

    def __init__(self, a, b, t, tau_net, micro_net, stop_net, start_net,
                 alpha_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b, a)  = P(a | b, s)
        self.stop_net = stop_net  # s -> 2b of (stop, 1 - stop) aka beta.
        self.start_net = start_net  # t -> b  aka P(b | t)
        self.alpha_net = alpha_net  # (t + b) -> t abstract transition

    def forward(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, t) tensor of abstract states t_i
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_NET_STOP_IX, STOP_NET_CONTINUE_IX
           (T, b) tensor of start logps
           (T, b, t, t) tensor of causal consistency penalties
        """
        T = s_i.shape[0]

        t_i = self.tau_net(s_i)  # (T, t)
        action_logps = self.micro_net(s_i).reshape(T, self.b, self.a)
        stop_logps = self.stop_net(s_i).reshape(T, self.b, 2)
        start_logps = self.start_net(t_i)  # (T, b) aka P(b | t)
        consistency_penalty = self.calc_consistency_penalty(t_i)  # (T, T, b)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return t_i, action_logps, stop_logps, start_logps, consistency_penalty

    def new_option_logps(self, t):
        """
        Input: an abstract state of shape (t,)
        Output: (b,) logp of different actions.
        """
        return self.start_net(t.unsqueeze(0)).reshape(self.b)

    def alpha_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        """
        return self.alpha_transitions(t.unsqueeze(0),
                                      torch.tensor([b])).reshape(self.t)

    def alpha_transitions(self, t_i, bs):
        """
        input: t_i: (T, t) batch of abstract states.
               bs: 1D tensor of actions to try
        returns: (T, |bs|, self.t) batch of new abstract states for each
            option applied.
        """
        # TODO: recalculate with einops, compare calculations
        T = t_i.shape[0]
        nb = bs.shape[0]
        # calculate transition for each t_i + b pair
        t_i2 = t_i.repeat_interleave(nb, dim=0)  # (T*nb, t)
        assertEqual(t_i2.shape, (T * nb, self.t))
        b_onehots = F.one_hot(bs, num_classes=self.b).repeat(T, 1)  # (T*nb, b)
        assertEqual(b_onehots.shape, (T * nb, self.b))
        # b is "less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*nb, t + b)
        assertEqual(t_i2.shape, (T * nb, self.t + self.b))
        # (T * nb, t + b) -> (T * nb, t)
        t_i2 = self.alpha_net(t_i2)
        return t_i2.reshape(T, nb, self.t)

    def calc_consistency_penalty(self, t_i):
        # TODO: recalculate with einops, compare calculations
        T = t_i.shape[0]
        # apply each action at each timestep.
        alpha_trans = self.alpha_transitions(t_i, torch.arange(self.b))
        alpha_trans = alpha_trans.reshape(T, 1, self.b, self.t)
        t_i2 = t_i.reshape(1, T, 1, self.t)
        # (start, end, action, t value)
        penalty = (t_i2 - alpha_trans)**2
        assertEqual(penalty.shape, (T, T, self.b, self.t))
        # L1 norm
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty


class Eq2Net(nn.Module):
    def __init__(self, abstract_policy_net, abstract_penalty=0.5,
                 consistency_ratio=1.):
        super().__init__()
        self.abstract_policy_net = abstract_policy_net
        self.a = abstract_policy_net.a
        self.b = abstract_policy_net.b
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

        At a high level:
            - keeps track of distribution over abstract actions, and what timestep
              that abstract action started. so (i+1, b) shape.
            - uses this to calculate expected
        """
        T = len(actions)
        assertEqual(s_i.shape[0], T + 1)
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
                stop_lps = stop_logps[i, :, STOP_NET_STOP_IX]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, STOP_NET_CONTINUE_IX]  # (b,)
                start_lps = start_logps[i]  # (b,)

                # prob mass for options exiting which started at step i; broadcast
                option_step_stops = option_step_dist + stop_lps.reshape(1, self.b)  # (i+1, b)
                total_rearrange = torch.logsumexp(option_step_stops, dim=(0, 1))
                total_rearrange = total_rearrange - self.abstract_penalty
                # distribute new mass among new options. broadcast
                new_mass = start_lps + total_rearrange  # (b,)

                # mass that stays in place, aka doesn't stop; broadcast
                option_step_dist = option_step_dist + one_minus_stop_lps  # (T, b)
                # add new mass at new timestep; TODO: einops?
                option_step_dist = torch.cat((option_step_dist,
                                             new_mass.unsqueeze(0)))

                # causal consistency penalty; start up to current timestep, end here,
                consistency_pens = consistency_penalties[:i + 1, i, :]  # (i+1, b)
                assertEqual(consistency_pens.shape, (i + 1, self.b))
                consistency_penalty = torch.logsumexp(option_step_dist + consistency_pens, dim=(0, 1))
                # TODO: this needs to be a logsumexp
                total_consistency_penalty += consistency_penalty

            action_lps = action_logps[i, :, action]  # (b,)
            # in prob space, this is a sum of probs weighted by macro-dist
            logp = torch.logsumexp(action_lps + option_step_dist, dim=(0, 1))
            # TODO: does this need to be a logsumexp?
            total_logp += logp

        # all macro options need to stop at the very end.
        final_stop_lps = stop_logps[-1, :, STOP_NET_STOP_IX]  # (b,)
        # broadcast
        total_logp += torch.logsumexp(final_stop_lps.reshape(1, self.b) + option_step_dist, dim=(0, 1))

        # maximize logp, minimize causal inconsistency
        loss = -total_logp + self.consistency_ratio * total_consistency_penalty
        return loss


def train_abstractions(data, net, epochs, lr=1E-3):
    print(f"net has {utils.num_params(net)} parameters")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print_every = epochs / 10

    net.train()

    dataloader = DataLoader(data, batch_size=256, shuffle=False)

    model_name = 'model_1-21.pt'
    try:
        for epoch in range(epochs):
            train_loss = 0
            start = time.time()
            for s_i_batch, actions_batch in dataloader:
                s_i_batch = s_i_batch.to(DEVICE)
                actions_batch = actions_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = net(s_i_batch, actions_batch)
                loss = torch.sum(loss)  # sum over batch
                train_loss += loss
                loss.backward()
                optimizer.step()

            if epoch % print_every == 0:
                print(f"epoch: {epoch}\t"
                      + f"train loss: {loss}\t"
                      + f"({time.time() - start:.0f}s)")

        utils.save_model(net, f'models/{model_name}')
    except KeyboardInterrupt:
        utils.save_model(net, f'models/{model_name}')


def train_supervised(dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        correct = 0
        total = 0
        for states, actions in dataloader:
            optimizer.zero_grad()
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            pred = net(states)
            loss = criterion(pred, actions)
            train_loss += loss
            loss.backward()
            optimizer.step()

            pred2 = torch.argmax(pred, dim=1)
            correct += sum(pred2 == actions).item()
            total += len(actions)

        acc = correct / total
        metrics = dict(
            epoch=epoch,
            loss=loss.item(),
            acc=acc,
        )
        mlflow.log_metrics(metrics, step=epoch)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {loss}\t"
                  + f"acc: {acc:.3f}\t"
                  + f"({time.time() - start:.1f}s)")
        if save_every and epoch % save_every == 0:
            utils.save_mlflow_model(net, model_name=f"epoch-{epoch}")


def sample_trajectories(net, data, full_abstract=False):
    """
    To sample with options:
    1. sample option from start state.
    2. choose actions according to option policy until stop.
    3. after stopping, sample new option.
    4. repeat until done.

    if full_abstract=True, then we will execute alpha(t_0, b) to get new t_i to
    sample new option from. Otherwise, will get t_i from tau(s_i).
    """
    for i in range(len(data.trajs)):
        points = data.points[i]
        (x, y, x_goal, y_goal) = points[0][0]
        moves = ''.join(data.trajs[i])
        print(f'({x, y}) to ({x_goal, y_goal}) via {moves}')
        moves_taken = ''
        options = []
        current_option_path = ''
        start_t_of_current_option = None
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
                start_t_of_current_option = t_i[0]
            else:
                # possibly stop previous option!
                stop = Categorical(logits=stop_logps[0, option, :]).sample()
                if stop == STOP_NET_STOP_IX:
                    if full_abstract:
                        new_t_i = net.abstract_policy_net.alpha_transition(start_t_of_current_option, option)
                        logits = net.abstract_policy_net.new_option_logps(new_t_i)
                        start_t_of_current_option = new_t_i
                    else:
                        logits = start_logps[0]

                    option = Categorical(logits=logits).sample()
                    options.append(current_option_path)
                    current_option_path = ''

            current_option_path += str(option.item())
            action = Categorical(logits=action_logps[0, option, :]).sample()
            # print(f"action: {action}")
            x, y = up_right.TrajData.execute((x, y), action)
            move = 'R' if action == 0 else 'U'
            moves_taken += move
            # print(f'now at ({x, y})')
        options.append(current_option_path)
        print(f'({0, 0}) to ({x, y}) via')
        print(f'{moves_taken}')
        print(f"{''.join(options)}")
        print('-' * 10)


def old_box_world_train():
    print('generating trajectories')

    env = old_box_world.make_env()
    trajs = old_box_world.generate_box_world_data(n=10, env=env)
    print('trajectories generated')
    data = old_box_world.BoxworldData(trajs)

    a = 4
    b = 5
    t = 10
    tau_net = AllConv(output_dim=t, input_filters=3)
    micro_net = AllConv(output_dim=b * a, input_filters=3)
    stop_net = AllConv(output_dim=b * 2, input_filters=3)
    start_net = FC(t, b, hidden_dim=128, num_hidden=2)
    alpha_net = FC(t + b, t, hidden_dim=64, num_hidden=1)

    abstract_policy_net = AbstractPolicyNet(
        a, b, t,
        tau_net,
        micro_net,
        stop_net,
        start_net,
        alpha_net)

    net = Eq2Net(abstract_policy_net,
                 abstract_penalty=0.001,
                 consistency_ratio=1.0)

    utils.load_model(net, f'models/model_12-2__20.pt')
    train_abstractions(data, net, epochs=1)
    # utils.save_model(net, f'models/model_12-2.pt')
    old_box_world.sample_trajectories(net,
                                      n=10,
                                      env=env,
                                      max_steps=data.max_steps + 10,
                                      full_abstract=True,
                                      render=True)


def box_world_sv_train(n=1000, epochs=100, drlnet=True, rounds=-1, num_test=100, test_every=1):
    print('New box world environment')
    mlflow.set_experiment("Boxworld sv train")
    with mlflow.start_run():
        env = box_world.BoxWorldEnv()

        # model_load_run_id = 'b8149a1f84f84edcb6ff0c389f80db78'
        model_load_run_id = None
        print_every = epochs / 5
        save_every = 1

        mlflow.log_params(dict(model_load_run_id=model_load_run_id,
                               drlnet=drlnet,
                               epochs=epochs,
                               ))
        if drlnet:
            net = RelationalDRLNet(input_channels=box_world.NUM_ASCII, num_attn_blocks=2, num_heads=4).to(DEVICE)
        else:
            net = AllConv(input_filters=box_world.NUM_ASCII,
                          residual_blocks=2,
                          residual_filters=24,
                          output_dim=4).to(DEVICE)
        if model_load_run_id is not None:
            utils.load_mlflow_model(net, model_load_run_id)

        print(f"Net has {utils.num_params(net)} parameters")

        try:
            round = 0
            while round != rounds:
                print(f'Round {round}')

                with Timing("Generated trajectories"):
                    data = box_world.BoxWorldDataset(env=env, n=n)

                print(f'{len(data)} examples')
                dataloader = DataLoader(data, batch_size=256, shuffle=True)

                train_supervised(dataloader, net, epochs=epochs, print_every=print_every)

                if round % test_every == 0:
                    with Timing("Evaluated model"):
                        box_world.eval_model(net, env, n=num_test)
                if round % save_every == 0:
                    utils.save_mlflow_model(net, overwrite=True)

                round += 1
                epochs = max(50, epochs - 25)
        except KeyboardInterrupt:
            utils.save_mlflow_model(net, overwrite=True)


def traj_box_world_sv_train(net, n=1000, epochs=100, rounds=-1, num_test=100, test_every=1):
    # does it HMM-style but without the abstract model
    mlflow.set_experiment("Boxworld sv train2 ")
    with mlflow.start_run():
        env = box_world.BoxWorldEnv()
        save_every = 1
        mlflow.log_params(dict(epochs=epochs))
        print(f"Net has {utils.num_params(net)} parameters")

        try:
            round = 0
            while round != rounds:
                print(f'Round {round}')
                env = box_world.BoxWorldEnv(seed=round)

                with Timing("Generated trajectories"):
                    data = box_world.BoxWorldTrajDataset(env=env, n=n)

                train_abstractions(data, net, epochs=epochs, lr=1E-4)

                if test_every and round % test_every == 0:
                    with Timing("Evaluated model"):
                        env = box_world.BoxWorldEnv(seed=round)
                        box_world.eval_model(net.abstract_policy_net.net, env, n=num_test)

                if round % save_every == 0:
                    utils.save_mlflow_model(net, overwrite=True)

                round += 1
                epochs = max(50, epochs - 25)
        except KeyboardInterrupt:
            utils.save_mlflow_model(net, overwrite=True)
