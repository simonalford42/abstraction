import torch.nn as nn
import planning
import data
import torch
import box_world
import utils
from einops import rearrange
from utils import assert_equal, assert_shape, DEVICE
from torch.utils.data import Dataset, DataLoader
from typing import Any
import wandb
from collections import Counter
import time


def fine_tune(control_net: nn.Module, params: dict[str, Any], data_net):
    if data_net == None:
        data_net = control_net

    env = box_world.BoxWorldEnv(seed=params.seed, solution_length=params.length)
    dataset = PlanningDataset(env, data_net, n=params.n, length=params.length[0])

    optimizer = torch.optim.Adam(control_net.parameters(), lr=params.lr)
    control_net.train()

    last_test_time = False
    last_save_time = time.time()
    epoch = 0
    updates = 0

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    if params.freeze_tau:
        control_net.tau_net.requires_grad_(False)  # tau(s)

    while updates < params.traj_updates:
        train_loss = 0

        for s0, actions in dataloader:
            s0, actions = s0.to(DEVICE), actions.to(DEVICE)
            B, T = actions.shape[0:2]

            loss = muzero_loss(s0, actions, control_net)

            train_loss += loss.item()
            loss = loss / (B * T)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            updates += len(s0)

        wandb.log({'loss': train_loss})

        if (params.test_every
                and (not last_test_time
                     or (time.time() - last_test_time > (params.test_every * 60)))):
            last_test_time = time.time()
            env = box_world.BoxWorldEnv(seed=params.seed, solution_length=params.length)
            utils.warn('fixed test env seed')
            print(f'Epoch {epoch}\ttrain_loss: {train_loss}')
            eval_planner(control_net, dataset)

        if (not params.no_log and params.save_every
                and (time.time() - last_save_time > (params.save_every * 60))):
            last_save_time = time.time()
            path = utils.save_model(control_net, f'models/{params.id}_control.pt')
            wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})

        epoch += 1

    if not params.no_log and params.save_every:
        path = utils.save_model(control_net, f'models/{params.id}_control.pt')
        wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})


nll_loss = nn.NLLLoss(reduction='none')


def muzero_loss(s0, options, control_net):
    '''
    s0: [B, *s] batch
    options: [B, T] batch
    '''
    B, T = options.shape
    assert_equal(s0.shape[0], B)

    t0 = control_net.tau_net(s0)
    t_i_preds = [t0]

    for i in range(T):
        b_i = options[:, i]
        t_i = t_i_preds[i]
        t_i_plus_one = control_net.macro_transitions2(t_i, b_i)
        t_i_preds.append(t_i_plus_one)

    t_i_preds = t_i_preds[:-1]  # don't need to predict action from end state
    t_i_batch = torch.stack(t_i_preds, dim=1)
    assert_shape(t_i_batch, (B, T, control_net.t))
    t_i_flattened = t_i_batch.reshape(B * T, control_net.t)
    option_logits = control_net.macro_policy_net(t_i_flattened).reshape(B, T, control_net.b)

    loss = nll_loss(rearrange(option_logits, 'B T b -> B b T'), options)
    assert_shape(loss, (B, T))
    return loss.sum()


def gen_planning_data(env, n, control_net, length):
    init_states = []
    all_options = []
    envs = []

    with torch.no_grad():
        while len(init_states) < n:
            env.reset()
            control_net.eval()
            env_copy = env.copy()
            solved, options, states_between_options = data.full_sample_solve(env.copy(), control_net, argmax=True, render=False)
            # print(f"options: {options}")
            control_net.train()

            if solved and len(options) == length:
                # move back to cpu so collation is ready
                s0 = states_between_options[0].cpu()
                # print(f's0 hash: {utils.hash_tensor(s0)}')
                init_states.append(s0)
                all_options.append(torch.tensor(options))
                envs.append(env_copy)

    return init_states, all_options, envs


def eval_planner(control_net, dataset):
    control_net.eval()

    total_solved = 0
    for options, env in zip(dataset.options, dataset.envs):
        solved, options2, _ = data.full_sample_solve(env.copy(), control_net, macro=True, argmax=True, render=False)
        print(f"options: {options}, options2: {options2} (solved={solved})")
        if solved:
            total_solved += 1

    print(f'Solved {total_solved}/{len(dataset)}')


class PlanningDataset(Dataset):
    def __init__(self, env, control_net: nn.Module, n, length: int):
        self.states, self.options, self.envs = gen_planning_data(env, n, control_net, length=length)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, ix):
        return self.states[ix], self.options[ix]


def main(control_net, params, data_net=None):
    fine_tune(control_net, params, data_net)


if __name__ == '__main__':
    main()
