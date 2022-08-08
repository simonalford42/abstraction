from typing import Any
import numpy as np
import wandb
import mlflow
from torch.utils.data import DataLoader
import planning
import argparse
import torch
import torch.nn as nn
import random
import utils
from utils import Timing, DEVICE, assert_shape, assert_equal
from abstract import boxworld_controller, boxworld_homocontroller
import abstract
from hmm import CausalNet, SVNet, HmmNet
import time
import box_world
import data
from modules import ShrinkingRelationalDRLNet, RelationalDRLNet
import muzero
import torch.nn.functional as F
import neurosym
from pyDatalog import pyDatalog as pyd


def fine_tune(control_net: nn.Module, params: dict[str, Any]):
    env = box_world.BoxWorldEnv(seed=params.seed)
    dataset = data.PlanningDataset(env, control_net, n=params.n, tau_precompute=params.tau_precompute)
    print(f'{len(dataset)} fine-tuning examples')
    params.epochs = int(params.traj_updates / len(dataset))

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                            collate_fn=data.latent_traj_collate)

    optimizer = torch.optim.Adam(control_net.parameters(), lr=params.lr)
    control_net.train()

    if params.tau_precompute:
        control_net.freeze_microcontroller()
    else:
        control_net.freeze_all_controllers()

    last_test_time = False
    last_save_time = time.time()
    epoch = 0
    updates = 0

    while updates < params.traj_updates:
        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle(batch_size=params.batch_size)

        train_loss = 0

        first = True
        for states, options, solveds, lengths, masks in dataloader:
            states, options, solveds, lengths, masks = [x.to(DEVICE) for x in [states, options, solveds, lengths, masks]]

            B, T, *s = states.shape

            if not params.tau_precompute:
                states_flattened = states.reshape(B * T, *s)
                t_i_flattened = control_net.tau_net(states_flattened)
                t_i = t_i_flattened.reshape(B, T, control_net.t)
            else:
                t_i = states

            # if params.test_every and epoch % params.test_every == 0 and first:
            #     preds = control_net.macro_transitions2(t_i[:, 0], options[:, 0])
            #     wandb.log({'avg cc loss': (t_i[:, 1] - preds).sum() / t_i.shape[0]})
            #     first = False

            loss = abstract.fine_tune_loss_v3(t_i, options, solveds, control_net, masks)

            train_loss += loss.item()
            loss = loss / sum(lengths)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            updates += B

        wandb.log({'loss': train_loss})

        if (params.test_every
                and (not last_test_time
                     or (time.time() - last_test_time > (params.test_every * 60)))):
            last_test_time = time.time()
            utils.warn('fixed test env seed')
            planning.eval_planner(
                control_net, box_world.BoxWorldEnv(seed=env.seed), n=params.num_test,
                # control_net, box_world.BoxWorldEnv(seed=env.seed + 1), n=params.num_test,
            )

        if (not params.no_log and params.save_every
                and (time.time() - last_save_time > (params.save_every * 60))):
            last_save_time = time.time()
            path = utils.save_model(control_net, f'models/{params.id}-epoch-{epoch}_control.pt')
            wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})

        epoch += 1

    if not params.no_log and params.save_every:
        path = utils.save_model(control_net, f'models/{params.id}_control.pt')
        wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})


def learn_options(net: nn.Module, params: dict[str, Any]):
    dataset = data.BoxWorldDataset(box_world.BoxWorldEnv(seed=params.seed), n=params.n, traj=True)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, collate_fn=data.traj_collate)

    params.epochs = int(params.traj_updates / params.n)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    net.train()
    test_env = box_world.BoxWorldEnv()
    print(f"Net has {utils.num_params(net)} parameters")

    last_test_time = False
    last_save_time = time.time()
    updates = 0
    epoch = 0

    while updates < params.traj_updates:
        if params.gumbel:
            abstract.GUMBEL_TEMP = params.gumbel_sched(epoch / params.epochs)

        if params.freeze is not False and updates / params.traj_updates >= params.freeze:
            # net.control_net.freeze_all_controllers()
            net.control_net.freeze_microcontroller()

        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle(batch_size=params.batch_size)

        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)

            loss = net(s_i_batch, actions_batch, lengths, masks)

            train_loss += loss.item()
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            loss.backward()
            optimizer.step()

        wandb.log({'loss': train_loss})
        if params.gumbel:
            wandb.log({'gumbel_temp': abstract.GUMBEL_TEMP})

        if (params.test_every
                and (not last_test_time
                     or (time.time() - last_test_time > (params.test_every * 60)))):
            last_test_time = time.time()

            # test_env = box_world.BoxWorldEnv(seed=params.seed)
            # print('fixed test env')
            test_acc = data.eval_options_model(net.control_net, test_env, n=params.num_test)
            # test_acc = planning.eval_planner(
            #     net.control_net, box_world.BoxWorldEnv(seed=params.seed), n=params.num_test,
            # )
            wandb.log({'test/acc': test_acc})

        if (not params.no_log and params.save_every
                and (time.time() - last_save_time > (params.save_every * 60))):
            last_save_time = time.time()
            path = utils.save_model(net, f'models/{params.id}-epoch-{epoch}.pt')
            wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})

        epoch += 1
        updates += params.n

    if not params.no_log:
        path = utils.save_model(net, f'models/{params.id}.pt')
        wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})


def sv_train(run, dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    """
    Train a basic supervised model, no option learning or anything.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)
            loss = net(s_i_batch, actions_batch, lengths, masks)
            if isinstance(net, ShrinkingRelationalDRLNet):
                loss += net.shrink_loss()

            train_loss += loss.item()
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            if run:
                run[f'{epoch}batch/loss'].log(loss.item())
                run[f'{epoch}batch/avg length'].log(sum(lengths) / len(lengths))
                run[f'{epoch}batch/mem'].log(utils.get_memory_usage())
            loss.backward()
            optimizer.step()

        if run:
            run['epoch'].log(epoch)
            run['loss'].log(train_loss)
            run['time'].log(time.time() - start)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")


def neurosym_symbolic_supervised_state_abstraction(dataloader: DataLoader, net, params):
    """
    Train a basic supervised model.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    net.train()

    params.epochs = int(params.traj_updates / params.n)

    updates = 0
    epoch = 0

    while updates < params.traj_updates:
        total_hard_matches = 0
        total_negatives = 0
        total_positives = 0
        total_negative_matches = 0
        total_positive_matches = 0

        train_loss = 0
        start = time.time()
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = net(inputs)
            # greedily convert probabilities to hard predictions
            hard_preds = torch.round(preds)
            # calculate number of hard matches by iterating through and counting perfect matches
            for pred, target in zip(hard_preds, targets):
                negative_spots = (target == 0)
                positive_spots = (target == 1)
                match_tensor = (target == pred)
                total_negatives += negative_spots.sum()
                total_positives += positive_spots.sum()
                total_negative_matches += (negative_spots * match_tensor).sum()
                total_positive_matches += (positive_spots * match_tensor).sum()

                if torch.equal(pred, target):
                    total_hard_matches += 1

            loss = F.binary_cross_entropy(preds, targets, reduction='mean')
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        acc = total_hard_matches / len(dataloader.dataset)
        negative_acc = total_negative_matches / total_negatives
        positive_acc = total_positive_matches / total_positives

        wandb.log({'loss': train_loss,
                   'acc': acc,
                   'negative_acc': negative_acc,
                   'positive_acc': positive_acc})

        epoch += 1
        updates += len(dataloader.dataset)

    if not params.no_log and params.save_every:
        path = utils.save_model(net, f'models/{params.id}_neurosym.pt')
        wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})


def learn_neurosym_world_model(dataloader: DataLoader, net, world_model_program, params):
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    net.train()

    params.epochs = int(params.traj_updates / params.n)

    updates = 0
    epoch = 0

    while updates < params.traj_updates:
        train_loss = 0
        count = 0
        for states, moves, target_states in dataloader:
            count += 1
            optimizer.zero_grad()

            states, moves, target_states = states.to(DEVICE), moves.to(DEVICE), target_states.to(DEVICE)
            state_embeds = net(states)
            target_state_embeds = net(target_states)
            state_preds = neurosym.world_model_step_prob(state_embeds, moves, world_model_program)
            # print(f"{state_preds.shape=}")
            # print(f"{target_state_embeds.shape=}")
            # print(f"{state_preds=}")
            # print(f"{target_state_embeds=}")
            # loss = F.binary_cross_entropy(state_preds.exp(), target_state_embeds.exp())
            loss = F.mse_loss(state_preds, target_state_embeds)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        wandb.log({'loss': train_loss})

        epoch += 1
        updates += len(dataloader.dataset)

    if not params.no_log and params.save_every:
        path = utils.save_model(net, f'models/{params.id}_neurosym.pt')
        wandb.log({'models': wandb.Table(columns=['path'], data=[[path]])})


def adjust_state_dict(state_dict):
    state_dict2 = {}
    for k, v in state_dict.items():
        if 'tau_net' in k:
            x = len('control_net.tau_net.')
            k2 = k[:x] + '0.' + k[x:]
            state_dict2[k2] = v
        else:
            state_dict2[k] = v
    return state_dict2


def make_gumbel_schedule_fn(params):
    if not params.gumbel:
        return False

    plateau_percent = 0.8
    # r is set so that np.exp(-r * plateau_percent) = 0.5
    r = -np.log(0.5) / plateau_percent

    def schedule_temp(percent_through):
        # between 0.5 and 1
        x = np.exp(-r * percent_through)
        # between 0 and 1
        x = 2 * x - 1
        # between stop and start
        x = params.g_stop_temp + x * (params.g_start_temp - params.g_stop_temp)
        return max(params.g_stop_temp, x)

    return schedule_temp


def make_net(params):
    if params.load:
        net = utils.load_model(params.model_load_path)
    elif params.model == 'sv':
        net = SVNet(boxworld_homocontroller(b=1, shrink_micro_net=params.shrink_micro_net), shrink_micro_net=params.shrink_micro_net)
    else:
        if params.model == 'hmm-homo':
            # HMM algorithm where start probs, stop probs, action probs all come from single NN
            control_net = boxworld_homocontroller(b=params.b)
        else:
            typ = 'hetero' if params.model in ['hmm', 'cc'] else params.model
            control_net = boxworld_controller(typ, params)
        if params.model in ['hmm', 'hmm-homo', 'ccts']:
            net = HmmNet(control_net, abstract_pen=params.abstract_pen, shrink_micro_net=params.shrink_micro_net)
        elif params.model == 'cc':
            assert not params.shrink_micro_net
            net = CausalNet(control_net, abstract_pen=params.abstract_pen)
        else:
            raise NotImplementedError()

    return net


def boxworld_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--traj_updates', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--b', type=int, default=10, help='number of options')
    parser.add_argument('--abstract_pen', type=float, default=1.0, help='for starting a new option, this penalty is subtracted from the overall logp of the seq')
    parser.add_argument('--model', type=str, default='cc', choices=['sv', 'cc', 'hmm-homo', 'hmm', 'ccts', 'ccts-reduced'])
    parser.add_argument('--seed', type=int, default=1, help='seed=0 chooses a random seed')
    parser.add_argument('--lr', type=float, default=8E-4)
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--abstract_dim', type=int, default=32)
    parser.add_argument('--tau_noise_std', type=float, default=0.0, help='STD of N(0, sigma) noise added to abstract state embedding to aid planning')
    parser.add_argument('--freeze', type=float, default=False, help='what % through training to freeze some subnets of control net')

    parser.add_argument('--load', action='store_true')
    parser.add_argument('--ellis', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--tau_precompute', action='store_true')
    parser.add_argument('--replace_trans_net', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--no_tau_norm', action='store_true')
    parser.add_argument('--relational_micro', action='store_true')
    parser.add_argument('--toy_test', action='store_true')
    parser.add_argument('--separate_option_nets', action='store_true')
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--g_start_temp', type=float, default=1)
    parser.add_argument('--g_stop_temp', type=float, default=1)
    parser.add_argument('--num_categories', type=int, default=8)
    parser.add_argument('--shrink_micro_net', action='store_true')
    parser.add_argument('--shrink_loss_scale', type=float, default=1)
    parser.add_argument('--length', type=int, default=(1, 2, 3, 4), choices=[1, 2, 3, 4],
                        help='box world env solution_length, may be single number or tuple of options')
    parser.add_argument('--muzero', action='store_true')
    parser.add_argument('--muzero_scratch', action='store_true')
    parser.add_argument('--num_test', type=int, default=200)
    parser.add_argument('--test_every', type=float, default=60, help='number of minutes to test every, if false will not test')
    parser.add_argument('--save_every', type=float, default=180, help='number of minutes to save every, if false will not save')
    parser.add_argument('--neurosym', action='store_true')
    params = parser.parse_args()

    featured_params = ['n', 'model', 'abstract_pen', 'fine_tune', 'muzero']

    if not params.seed:
        seed = random.randint(0, 2**32 - 1)
        params.seed = seed

    random.seed(params.seed)
    torch.manual_seed(params.seed)

    if type(params.length) == int:
        params.length = (params.length, )  # box_world env expects tuple

    if not hasattr(params, 'batch_size'):
        if params.relational_micro or params.gumbel or params.shrink_micro_net:
            params.batch_size = 16
        else:
            params.batch_size = 32
        if params.ellis:  # more memory available!
            params.batch_size *= 2

    if params.fine_tune and not params.load:
        print('WARNING: params.load = False, creating new model')
        # params.load = True

    if not hasattr(params, 'traj_updates'):
        params.traj_updates = 1E8 if (params.fine_tune or params.muzero or params.neurosym) else 1E7  # default: 1E7
    params.model_load_path = 'models/e14b78d01cc548239ffd57286e59e819.pt'
    params.gumbel_sched = make_gumbel_schedule_fn(params)
    params.device = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'cpu'

    if params.toy_test:
        params.n = 100
        params.traj_updates = 1000
        params.test_every = 1
        params.save_every = False
        params.no_log = True
        params.num_test = 5

    if params.no_log:
        global mlflow
        mlflow = utils.NoMlflowRun()
    else:
        mlflow.set_experiment('Boxworld 3/22')

    if params.muzero:
        params.load = True

    net = make_net(params).to(DEVICE)

    if params.muzero:
        data_net = net

        if params.muzero_scratch:
            params.load = False
            net = make_net(params).to(DEVICE)

    with Timing('Completed training'):
        with mlflow.start_run():
            params.id = mlflow.active_run().info.run_id
            print(f"Starting run:\n{mlflow.active_run().info.run_id}")

            for p in featured_params:
                print(p.upper() + ': \t' + str(getattr(params, p)))
            print(f'{params=}')

            wandb.init(project="abstraction",
                       mode='disabled' if params.no_log else 'online',
                       config=vars(params))

            if params.muzero:
                muzero.main(net.control_net, params, data_net=data_net.control_net)
            elif params.fine_tune:
                fine_tune(net.control_net, params)
            elif params.neurosym:
                neurosym_train(params)
            else:
                learn_options(net, params)

            for p in featured_params:
                print(p.upper() + ': \t' + str(getattr(params, p)))


def neurosym_train(params):
    # seems like I have to do this outside of the function to get it to work?
    pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')

    env = box_world.BoxWorldEnv(solution_length=(4, ), num_forward=(4, ))

    # abs_data = neurosym.ListDataset(neurosym.supervised_symbolic_state_abstraction_data(env, n=params.n))
    abs_data = neurosym.ListDataset(neurosym.world_model_data(env, n=params.n))
    dataloader = DataLoader(abs_data, batch_size=params.batch_size, shuffle=True)

    net = neurosym.AbstractEmbedNet(neurosym.RelationalDRLNet(input_channels=box_world.NUM_ASCII, out_dim=2 * 2 * box_world.NUM_COLORS * box_world.NUM_COLORS))
    print(net.out_dim)
    assert False
    net = net.to(DEVICE)
    print(f"Net has {utils.num_params(net)} parameters")
    # neurosym_symbolic_supervised_state_abstraction(dataloader, net, params)
    learn_neurosym_world_model(dataloader, net, neurosym.BW_WORLD_MODEL_PROGRAM, params)


if __name__ == '__main__':
    boxworld_main()
