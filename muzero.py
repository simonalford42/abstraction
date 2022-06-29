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


def fine_tune(control_net: nn.Module, data, params: dict[str, Any]):
    optimizer = torch.optim.Adam(control_net.parameters(), lr=params.lr)
    control_net.train()

    epoch = 0
    updates = 0

    dataloader = DataLoader(data, batch_size=params.batch_size, shuffle=True)

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

        if params.test_every and epoch % (max(1, params.test_every // 5)) == 0:
            print(f"train_loss: {train_loss}")

        if params.test_every and epoch % params.test_every == 0:
            env = box_world.BoxWorldEnv(seed=params.seed, solution_length=params.length)
            utils.warn('fixed test env seed')
            print(f'Epoch {epoch}')
            planning.eval_planner(control_net, env, n=params.num_test)

        if not params.no_log and params.save_every and epoch % params.save_every == 0 and epoch > 0:
            utils.save_model(control_net, f'models/{params.id}_control.pt')

        epoch += 1

    if not params.no_log:
        utils.save_model(control_net, f'models/{params.id}_control.pt')


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


def gen_planning_data2(env, n, control_net, length):
    init_states = []
    all_options = []

    with torch.no_grad():
        while len(init_states) < n:
            env.reset()
            control_net.eval()
            solved, options, states_between_options = data.full_sample_solve(env.copy(), control_net, argmax=True, render=False)
            print(f"options: {options}")
            control_net.train()

            if len(options) == length:
                # move back to cpu so collation is ready
                s0 = states_between_options[0].cpu()
                init_states.append(s0)
                all_options.append(torch.tensor(options))

    return init_states, all_options


class PlanningDataset2(Dataset):
    def __init__(self, env, control_net: nn.Module, n, length: int):
        self.states, self.options = gen_planning_data2(env, n, control_net, length=length)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, ix):
        return self.states[ix], self.options[ix]


def boxworld_main(env, control_net, params):
    print('muzero')
    env = box_world.BoxWorldEnv(seed=params.seed, solution_length=params.length)
    data = PlanningDataset2(env, control_net, n=params.n, length=params.length[0])
    fine_tune(control_net, data, params)


if __name__ == '__main__':
    boxworld_main()
