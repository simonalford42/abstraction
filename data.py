from typing import List, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import utils
from utils import assert_equal, DEVICE
from torch.distributions import Categorical
import torch.nn as nn
import box_world as bw
import random
import wandb
import neurosym


STOP_IX = 0
CONTINUE_IX = 1 - STOP_IX
UNSOLVED_IX, SOLVED_IX = 0, 1


def eval_options_model_interactive(control_net, env, n=100, option='silent'):
    control_net.eval()
    num_solved = 0
    check_cc = hasattr(control_net, 'tau_net')
    check_solved = hasattr(control_net, 'solved_net')
    cc_losses = []
    correct_solved_preds = 0

    for i in range(n):
        env2 = env.copy()
        eval_options_model(control_net, env2, n=1, option='verbose')
        obs = env.reset()
        options_trace = obs
        option_map = {i: [] for i in range(control_net.b)}
        done, solved = False, False
        correct_solved_pred = True
        t = -1
        options = []
        moves_without_moving = 0
        prev_pos = (-1, -1)

        current_option = None
        tau_goal = None

        while not (done or solved):
            t += 1
            bw.render_obs(obs, pause=0.1)
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            # (b, a), (b, 2), (b, ), (2, )
            action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs, option_start_s=obs)

            if check_solved:
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX
                # if verbose:
                #     print(f'solved prob: {torch.exp(solved_logits[SOLVED_IX])}')
                #     print(f'is_solved_pred: {is_solved_pred}')
                if is_solved_pred:
                    correct_solved_pred = False

            if current_option is not None:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
            new_option = current_option is None or stop == STOP_IX
            if new_option:
                if check_cc:
                    tau = control_net.tau_embed(obs)
                if current_option is not None:
                    if check_cc:
                        cc_loss = ((tau_goal - tau)**2).sum()
                        print(f'cc_loss: {cc_loss}')
                        cc_losses.append(cc_loss.item())
                    options_trace[prev_pos] = 'e'
                print(f'Choose new option b; start probs = {torch.exp(start_logps)}')
                b = int(input('b='))
                # current_option = Categorical(logits=start_logps).sample().item()
                current_option = b
                option_start_s = obs
                if check_cc:
                    tau_goal = control_net.macro_transition(tau, current_option)
            else:
                # dont overwrite red dot
                if options_trace[prev_pos] != 'e':
                    options_trace[prev_pos] = 'm'

            options.append(current_option)

            a = Categorical(logits=action_logps[current_option]).sample().item()
            option_map[current_option].append(a)

            obs, rew, done, info = env.step(a)
            solved = env.solved

            pos = bw.player_pos(obs)
            if prev_pos == pos:
                moves_without_moving += 1
            else:
                moves_without_moving = 0
                prev_pos = pos
            if moves_without_moving >= 5:
                done = True

        if solved:
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)

            if check_solved:
                # check that we predicted that we solved
                _, _, _, solved_logits = control_net.eval_obs(obs, option_start_s)
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX

                if not is_solved_pred:
                    correct_solved_pred = False
                else:
                    if correct_solved_pred:
                        correct_solved_preds += 1

            # add cc loss from last action.
            if check_cc:
                tau = control_net.tau_embed(obs)
                cc_loss = ((tau_goal - tau)**2).sum()
                cc_losses.append(cc_loss.item())
            num_solved += 1

    if check_cc:
        cc_loss_avg = sum(cc_losses) / len(cc_losses)
    if check_solved:
        solved_acc = 0 if not num_solved else correct_solved_preds / num_solved
        # print(f'Correct solved pred: {solved_acc:.2f}')

    control_net.train()
    if check_cc:
        print(f'Solved {num_solved}/{n} episodes, CC loss avg = {cc_loss_avg}')
    else:
        print(f'Solved {num_solved}/{n} episodes')
    return num_solved / n


def eval_options_model(control_net, env, n=100, render=False, run=None, epoch=None, argmax=True, symbolic_print=False):
    """
    control_net needs to have fn eval_obs that takes in a single observation,
    and outputs tuple of:
        (b, a) action logps
j       (b, 2) stop logps
        (b, ) start logps

    """
    control_net.eval()
    num_solved = 0
    check_cc = hasattr(control_net, 'tau_net')
    check_solved = hasattr(control_net, 'solved_net')
    cc_losses = []
    correct_solved_preds = 0

    for i in range(n):
        video_obss = []
        obs = env.reset()
        if run and i < 10:
            run[f'test/epoch {epoch}/obs'].log(bw.obs_figure(obs), name='obs')
        options_trace = obs
        option_map = {i: [] for i in range(control_net.b)}
        done, solved = False, False
        correct_solved_pred = True
        t = -1
        options = []
        moves_without_moving = 0
        prev_pos = (-1, -1)

        current_option = None
        tau_goal = None
        done_video = False

        while not (done or solved):
            t += 1
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            # (b, a), (b, 2), (b, ), (2, )
            action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs, option_start_s=obs)

            if check_solved:
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX
                if is_solved_pred:
                    correct_solved_pred = False

            if current_option is not None:
                if argmax:
                    stop = torch.argmax(stop_logps[current_option]).item()
                else:
                    stop = Categorical(logits=stop_logps[current_option]).sample().item()
                if stop == STOP_IX:
                    done_video = True
            new_option = current_option is None or stop == STOP_IX
            if new_option:
                if check_cc:
                    tau = control_net.tau_embed(obs)
                if current_option is not None:
                    if check_cc:
                        cc_loss = ((tau_goal - tau)**2).sum()
                        cc_losses.append(cc_loss.item())
                    options_trace[prev_pos] = 'e'
                if argmax:
                    current_option = torch.argmax(start_logps).item()
                else:
                    current_option = Categorical(logits=start_logps).sample().item()
                option_start_s = obs
                if check_cc:
                    tau_goal = control_net.macro_transition(tau, current_option)
            else:
                # dont overwrite red dot
                if options_trace[prev_pos] != 'e':
                    options_trace[prev_pos] = 'm'

            if symbolic_print and new_option:
                tau = control_net.tau_embed(obs)
                tau = rearrange(tau, '(p c1 c2 two) -> p c1 c2 two', p=2, c1=bw.NUM_COLORS, c2=bw.NUM_COLORS, two=2)
                tau = tau[:, :, :, neurosym.STATE_EMBED_TRUE_IX]
                held_keys, dominos = neurosym.parse_symbolic_tensor2(tau)
                held_keys = [(k, p) for (k, p) in held_keys if p > 1E-5]
                dominos = [(k, p) for (k, p) in dominos if p > 1E-5]
                held_keys = [f'{k}: {p:.7f}' for (k, p) in held_keys]
                dominos = [f'{k}: {p:.7f}' for (k, p) in dominos]
                print(held_keys[:min(3, len(held_keys))], dominos[:min(5, len(dominos))])

            options.append(current_option)

            if argmax:
                a = torch.argmax(action_logps[current_option]).item()
            else:
                a = Categorical(logits=action_logps[current_option]).sample().item()
            option_map[current_option].append(a)

            obs, rew, done, info = env.step(a)
            solved = env.solved

            pos = bw.player_pos(obs)
            if prev_pos == pos:
                moves_without_moving += 1
            else:
                moves_without_moving = 0
                prev_pos = pos
            if moves_without_moving >= 5:
                done = True

            if render:
                title = f'Executing option {current_option}'
                pause = 1.6 if new_option else 0.1
                # pause = 0.2
                if new_option:
                    title = f'Starting new option: {current_option}'
                option_map[current_option].append((obs, title, pause))
                bw.render_obs(obs, title=title, pause=pause)
                if not done_video:
                    video_obss.append((obs, title, pause))

        if render:
            if solved:
                title = 'Solved episode'
            else:
                title = 'Episode terminated (did not solve)'
            bw.render_obs(obs, title=title, pause=1)
            # video_obss.append((obs, title, 2))
            # animate.save_video(video_obss, f'new_video{i}')

        if solved:
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)

            if check_solved:
                # check that we predicted that we solved
                _, _, _, solved_logits = control_net.eval_obs(obs, option_start_s)
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX

                if not is_solved_pred:
                    correct_solved_pred = False
                else:
                    if correct_solved_pred:
                        correct_solved_preds += 1

            # add cc loss from last action.
            if check_cc:
                tau = control_net.tau_embed(obs)
                cc_loss = ((tau_goal - tau)**2).sum()
                cc_losses.append(cc_loss.item())
            num_solved += 1

        if run and i < 10:
            run[f'test/epoch {epoch}/obs'].log(bw.obs_figure(options_trace),
                                               name='orange=new option')

    if check_cc and len(cc_losses) > 0:
        cc_loss_avg = sum(cc_losses) / len(cc_losses)
        if run:
            run[f'test/cc loss avg'].log(cc_loss_avg)
    if check_solved:
        solved_acc = 0 if not num_solved else correct_solved_preds / num_solved
        if run:
            run[f'test/solved pred acc'].log(solved_acc)

    control_net.train()
    if check_cc and len(cc_losses) > 0:
        print(f'Solved {num_solved}/{n} episodes, CC loss avg = {cc_loss_avg}')
    else:
        print(f'Solved {num_solved}/{n} episodes')
    return num_solved / n


def eval_model(net, env, n=100, renderer: Callable = None):
    """
    renderer is a callable that takes in obs.
    """
    net.eval()
    num_solved = 0

    for i in range(n):
        obs = env.reset()
        done, solved = False, False
        t = 0

        while not (done or solved):
            t += 1
            if renderer is not None:
                renderer(obs)
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            action_logps, _, _ = net.eval_obs(obs)
            action_logps = action_logps[0]
            a = Categorical(logits=action_logps).sample().item()
            obs, rew, done, info = env.step(a)
            solved = env.solved

        if solved:
            num_solved += 1

    print(f'Solved {num_solved}/{n} episodes')
    net.train()
    return num_solved/n


def obs_to_tensor(obs) -> torch.Tensor:
    obs = torch.tensor([[bw.ascii_to_int(a) for a in row]
                       for row in obs])
    obs = F.one_hot(obs, num_classes=bw.NUM_ASCII).to(torch.float)
    assert_equal(obs.shape[-1], bw.NUM_ASCII)
    obs = rearrange(obs, 'h w c -> c h w')
    return obs


def tensor_to_obs(obs):
    obs = rearrange(obs, 'c h w -> h w c')
    obs = torch.argmax(obs, dim=-1)
    obs = np.array([[bw.int_to_ascii(i) for i in row]
                     for row in obs])
    return obs


def latent_traj_collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    '''
    takes (states, options, solveds) as input
    states can be either abstract or microscopic.

    for a trajectory with t_i of shape (T + 1, t) and b_i of shape (T, ),

    mask is a (max_T_plus_one, ) vector with mask[0:T+1] = 1

    Note that this is different from traj_collate
    '''
    # NOTE: different lengths
    lengths = torch.tensor([len(states) for states, _, _ in batch])
    max_T_plus_one = max(lengths)
    states_batch = []
    options_batch = []
    solveds_batch = []
    masks = []
    for states, options, solved in batch:
        T_plus_one, *s = states.shape
        to_add = max_T_plus_one - T_plus_one
        states2 = torch.cat((states, torch.zeros((to_add, *s))))
        options2 = torch.cat((options, torch.zeros(to_add, dtype=int)))
        # NOTE: different masking from traj_collate!!!
        mask = torch.zeros(max_T_plus_one, dtype=int)
        mask[:T_plus_one] = 1
        masks.append(mask)
        assert_equal(states2.shape, (max_T_plus_one, *s))
        assert_equal(options2.shape, (max_T_plus_one - 1, ))
        assert_equal(mask.shape, (max_T_plus_one, ))
        states_batch.append(states2)
        options_batch.append(options2)

        solveds = torch.zeros(max_T_plus_one, dtype=int)
        solveds[T_plus_one - 1] = solved
        solveds_batch.append(solveds)

    return torch.stack(states_batch), torch.stack(options_batch), torch.stack(solveds_batch), lengths, torch.stack(masks)


def traj_collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    """
    batch is a list of (states, moves, length, masks) tuples.
    """
    max_T = max([length for _, _, length in batch])
    # max_T = MAX_LEN
    states_batch = []
    moves_batch = []
    lengths = []
    masks = []
    for states, moves, length in batch:
        _, *s = states.shape
        T = moves.shape[0]
        to_add = max_T - T
        states2 = torch.cat((states, torch.zeros((to_add, *s))))
        moves2 = torch.cat((moves, torch.zeros(to_add, dtype=int)))
        assert_equal(states2.shape, (max_T + 1, *s))
        assert_equal(moves2.shape, (max_T, ))
        states_batch.append(states2)
        moves_batch.append(moves2)
        lengths.append(length)
        mask = torch.zeros(max_T, dtype=int)
        mask[:T] = 1
        masks.append(mask)

    return torch.stack(states_batch), torch.stack(moves_batch), torch.tensor(lengths), torch.stack(masks)


def bw_dataloader(env: bw.BoxWorldEnv, n: int, traj: bool = True, batch_size: int = 256):
    data = BoxWorldDataset(env, n, traj)
    if traj:
        return DataLoader(data, batch_size=batch_size, shuffle=not traj, collate_fn=traj_collate)
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=not traj)


def calc_latents(dataloader, control_net):
    all_t_i, all_b_i = [], []

    for s_i_batch, actions_batch, lengths, masks in dataloader:
        s_i_batch, actions_batch, lengths, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), lengths.to(DEVICE), masks.to(DEVICE)
        # (B, max_T+1, b, n), (B, max_T+1, b, 2), (B, max_T+1, b), (B, max_T+1, max_T+1, b), (B, max_T+1, 2)
        action_logps, stop_logps, start_logps, causal_pens, solved, traj_t_i = control_net(s_i_batch, batched=True, tau_noise=False)
        for batch_ix, length in enumerate(lengths):
            i = 0
            t_i = []
            b_i = []
            current_option = None
            while i < length - 1:
                if current_option is not None:
                    stop = Categorical(logits=stop_logps[batch_ix, i, current_option]).sample().item()
                if current_option is None or stop == STOP_IX:
                    current_option = Categorical(logits=start_logps[batch_ix, i]).sample().item()
                    b_i.append(current_option)
                    t_i.append(traj_t_i[batch_ix, i])
                i += 1
            # stop at end
            t_i.append(traj_t_i[batch_ix, length-1])
            all_t_i.append(torch.stack(t_i).cpu())
            all_b_i.append(torch.tensor(b_i).cpu())

    return all_t_i, all_b_i


def gen_planning_data(env, n, control_net, tau_precompute=False):
    all_states = []
    all_options = []
    all_solveds = []

    env = env.copy()
    total = 0
    with torch.no_grad():
        while total < n:
            env.reset()

            control_net.eval()

            out_dict = full_sample_solve(env.copy(), control_net, argmax=True, render=False)
            solved = out_dict['solved']
            options = out_dict['options']
            states_between_options = out_dict['states_between_options']


            control_net.train()

            # if not solved:
            #     continue

            total += 1

            states = torch.stack(states_between_options)
            if tau_precompute:
                states = control_net.tau_net(states)

            # move back to cpu so collation is ready
            states = states.cpu()
            all_states.append(states)
            all_options.append(torch.tensor(options))
            all_solveds.append(torch.tensor(int(solved)))

    return all_states, all_options, all_solveds


class PlanningDataset(Dataset):
    def __init__(self, env, control_net: nn.Module, n, tau_precompute=False):
        states, options, solveds = gen_planning_data(env, n, control_net, tau_precompute)

        states, options, solveds = zip(*sorted(zip(states, options, solveds),
                                               key=lambda t: t[0].shape[0]))
        self.states, self.options, self.solveds = [list(x) for x in [states, options, solveds]]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, ix):
        return self.states[ix], self.options[ix], self.solveds[ix]

    def shuffle(self, batch_size):
        """
        Shuffle trajs which share the same length, while still keeping the overall order the same.
        Then shuffles among the batches, so that the order in which lengths are encountered differs.
        """
        ixs = list(range(len(self.states)))
        random.shuffle(ixs)
        self.states[:] = [self.states[i] for i in ixs]
        self.options[:] = [self.options[i] for i in ixs]
        self.solveds[:] = [self.solveds[i] for i in ixs]
        self.states[:], self.options[:], self.solveds[:] = zip(*sorted(zip(self.states, self.options, self.solveds),
                                                                       key=lambda t: t[0].shape[0]))
        self.states, self.options, self.solveds = [list(x) for x in [self.states, self.options, self.solveds]]

        # keep the last batch at the end
        n = len(self.states)
        n = n - (n % batch_size)
        ixs = list(range(n))
        blocks = [ixs[i:i + batch_size] for i in range(0, n, batch_size)]
        random.shuffle(blocks)
        ixs = [b for bs in blocks for b in bs]
        assert_equal(len(ixs), n)
        self.states[:n] = [self.states[i] for i in ixs]
        self.options[:n] = [self.options[i] for i in ixs]
        self.solveds[:n] = [self.solveds[i] for i in ixs]


class BoxWorldDataset(Dataset):
    def __init__(self, env: bw.BoxWorldEnv, n: int, traj: bool = True, shuffle: bool = True):
        """
        If traj is true, spits out a trajectory and its actions.
        Otherwise, spits out a single state and its action.
        """
        # all in memory
        # list of (states, moves) tuple
        self.data: List[Tuple[List, List]] = [bw.generate_traj(env) for i in range(n)]
        # states, moves = self.data[0]
        # self.data = [(states[0:2], moves[0:1])]
        self.traj = traj

        # ignore last state
        self.states = [obs_to_tensor(s)
                       for states, _ in self.data for s in states[:-1]]
        self.moves = [torch.tensor(m) for _, moves in self.data for m in moves]
        assert_equal(len(self.states), len(self.moves))

        self.traj_states = [torch.stack([obs_to_tensor(s) for s in states]) for states, _ in self.data]
        self.traj_moves = [torch.stack([torch.tensor(m) for m in moves]) for _, moves in self.data]

        if shuffle:
            self.traj_states, self.traj_moves = zip(*sorted(zip(self.traj_states, self.traj_moves),
                                                    key=lambda t: t[0].shape[0]))
            self.traj_states, self.traj_moves = list(self.traj_states), list(self.traj_moves)
        assert_equal([m.shape[0] + 1 for m in self.traj_moves], [ts.shape[0] for ts in self.traj_states])

    def __len__(self):
        if self.traj:
            return len(self.traj_states)
        else:
            return len(self.states)

    def __getitem__(self, i):
        if self.traj:
            return self.traj_states[i], self.traj_moves[i], len(self.traj_moves[i])
        else:
            return self.states[i], self.moves[i]

    def shuffle(self, batch_size):
        """
        Shuffle trajs which share the same length, while still keeping the overall order the same.
        Then shuffles among the batches, so that the order in which lengths are encountered differs.
        """
        ixs = list(range(len(self.traj_states)))
        random.shuffle(ixs)
        self.traj_states[:] = [self.traj_states[i] for i in ixs]
        self.traj_moves[:] = [self.traj_moves[i] for i in ixs]
        self.traj_states[:], self.traj_moves[:] = zip(*sorted(zip(self.traj_states, self.traj_moves),
                                                      key=lambda t: t[0].shape[0]))
        self.traj_states, self.traj_moves = list(self.traj_states), list(self.traj_moves)

        # keep the last batch at the end
        n = len(self.traj_states)
        n = n - (n % batch_size)
        ixs = list(range(n))
        blocks = [ixs[i:i + batch_size] for i in range(0, n, batch_size)]
        random.shuffle(blocks)
        ixs = [b for bs in blocks for b in bs]
        assert_equal(len(ixs), n)
        self.traj_states[:n] = [self.traj_states[i] for i in ixs]
        self.traj_moves[:n] = [self.traj_moves[i] for i in ixs]


def full_sample_solve(env, control_net, render=False, macro=False, argmax=True):
    """
    macro: use macro transition model to base next option from previous trnasition prediction, to test abstract transition model.
    argmax: select options, actions, etc by argmax not by sampling.
    """
    control_net.eval()

    obs = env.obs

    options_trace = obs  # as we move, we color over squares in this where we moved, to render later
    done, solved = False, False
    option_at_step_i = []  # option at step i
    options = []
    moves_without_moving = 0
    prev_pos = (-1, -1)
    op_new_tau = None
    # op_new_tau_solved_prob = None
    moves = []
    states_between_options = []
    states_for_each_option = []
    moves_for_each_option = []
    option_states = []
    option_moves = []

    current_option = None

    while not (done or solved):
        obs = obs_to_tensor(obs).to(DEVICE)
        # (b, a), (b, 2), (b, ), (2, )
        action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs)

        if current_option is not None:
            if argmax:
                stop = torch.argmax(stop_logps[current_option]).item()
            else:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
        new_option = current_option is None or stop == STOP_IX
        if new_option:
            if current_option is not None:
                states_for_each_option.append(option_states)
                moves_for_each_option.append(option_moves)
                option_states = []
                option_moves = []

            states_between_options.append(obs)  # starts out empty, adds before each option, then adds final at end
            if current_option is not None and macro:
                start_logps = control_net.macro_policy_net(op_new_tau.unsqueeze(0))[0]

            tau = control_net.tau_embed(obs)
            if macro and op_new_tau is not None:
                tau = op_new_tau

            if current_option is not None:
                # causal_consistency = ((tau - op_new_tau)**2).sum()
                # print(f'causal_consistency: {causal_consistency}')
                options_trace[prev_pos] = 'e'

            if argmax:
                current_option = torch.argmax(start_logps).item()
            else:
                current_option = Categorical(logits=start_logps).sample().item()

            op_start_logps, op_new_taus, op_solved_logps = control_net.eval_abstract_policy(tau)
            op_new_tau = op_new_taus[current_option]
            # op_new_tau_solved_prob = torch.exp(op_solved_logps[current_option, SOLVED_IX])
            # print(f'solved prob from option: {op_new_tau_solved_prob}')
            options.append(current_option)
        else:
            # dont overwrite 'new option' dot from earlier
            if options_trace[prev_pos] != 'e':
                options_trace[prev_pos] = 'm'

        option_at_step_i.append(current_option)

        if argmax:
            a = torch.argmax(action_logps[current_option]).item()
        else:
            a = Categorical(logits=action_logps[current_option]).sample().item()
        moves.append(a)

        option_moves.append(a)
        option_states.append(obs)

        obs, rew, done, _ = env.step(a)

        if render:
            title = f'option={current_option}'
            pause = 1 if new_option else 0.01
            if new_option:
                title += f' (new option = {current_option})'
            bw.render_obs(obs, title=title, pause=pause)

        solved = env.solved

        pos = bw.player_pos(obs)
        if prev_pos == pos:
            moves_without_moving += 1
        else:
            moves_without_moving = 0
            prev_pos = pos
        if moves_without_moving >= 5:
            done = True

    obs = obs_to_tensor(obs).to(DEVICE)
    states_between_options.append(obs)

    states_for_each_option.append(option_states)
    moves_for_each_option.append(option_moves)

    # if solved:
    #     check that we predicted that we solved
    #     _, _, _, solved_logits = control_net.eval_obs(obs)
    #     print(f'END solved prob: {torch.exp(solved_logits[SOLVED_IX])}')

    if render:
        bw.render_obs(options_trace, title=f'{solved=}', pause=1 if solved else 3)

    assert len(options) == len(states_for_each_option)
    control_net.train()

    return {'solved': solved,
            'options': options,
            'states_between_options': states_between_options,
            'states_for_each_option': states_for_each_option,
            'moves_for_each_option': moves_for_each_option}


class ListDataset(Dataset):
    '''
    Generic Dataset class for data stored in a list.
    '''
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]


def sv_micro_data(n, typ='full_traj', control_net=None):
    '''
    typ: 'full_traj', 'from_model', 'ground_truth'
    '''

    env = bw.BoxWorldEnv()
    if typ == 'full_traj':
        # get list of (state, action) pairs from n trajectories
        data = BoxWorldDataset(env, n=n, traj=False)
        states, moves = data.states, data.moves
        options = [0] * len(states)
    elif typ == 'from_model':

        states, moves, options = [], [], []
        num_solved = 0
        for i in range(n):
            env.reset()
            out_dict = full_sample_solve(env, control_net)
            states_for_each_option = out_dict['states_for_each_option']
            moves_for_each_option = out_dict['moves_for_each_option']
            options1 = out_dict['options']
            num_solved += out_dict['solved']
            assert len(options1) == len(moves_for_each_option)
            for option_states, option_moves, option in zip(states_for_each_option, moves_for_each_option, options1):
                states.extend(option_states)
                moves.extend(option_moves)
                options.extend([option] * len(option_moves))
        print(f"Solved {num_solved}/{n} for data creation")

    else:
        assert typ == 'ground_truth'
        states, moves, options = [], [], []
        for i in range(n):
            traj_states, traj_moves, traj_options = bw.generate_traj_with_options(env)
            traj_states = [obs_to_tensor(s).to(DEVICE) for s in traj_states]
            states.extend(traj_states)
            moves.extend(traj_moves)
            options.extend(traj_options)

    return ListDataset(list(zip(states, moves, options)))
