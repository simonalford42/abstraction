from typing import Any
import numpy as np
import wandb
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
from modules import RelationalDRLNet, MicroNet2, MicroNet3
import muzero
import torch.nn.functional as F
import neurosym
# from pyDatalog import pyDatalog as pyd
from itertools import chain
from einops.layers.torch import Rearrange
from einops import rearrange
import os


def fine_tune_rnn(control_net, params):
    nll_loss = nn.NLLLoss(reduction='none')

    if params.load_rnn:
        rnn_model_id = '62f87e8a7da34f5fa84cd7408e84ca54-epoch-21826_rnn'
        rnn = utils.load_model(f'models/{rnn_model_id}.pt').to(DEVICE)
    else:
        rnn = torch.nn.GRU(input_size=params.b, hidden_size=params.abstract_dim, batch_first=True).to(DEVICE)

    env = box_world.BoxWorldEnv(seed=params.seed)

    dataset = data.PlanningDataset(env, control_net, n=params.n, tau_precompute=False)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                            collate_fn=data.latent_traj_collate)
    eval_dataset = data.PlanningDataset(env, control_net, n=200, tau_precompute=False)
    eval_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                            collate_fn=data.latent_traj_collate)

    params.epochs = int(params.traj_updates / len(dataset))

    optimizer = torch.optim.Adam(chain(rnn.parameters(), control_net.parameters()), lr=params.lr)

    last_save_time = time.time()
    epoch = 0
    updates = 0

    while updates < params.traj_updates:
        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle(batch_size=params.batch_size)

        train_loss = 0

        option_num_correct = torch.zeros(dataset.max_T).to(DEVICE)
        solved_num_correct = torch.zeros(dataset.max_T + 1).to(DEVICE)
        option_num_total = torch.zeros(dataset.max_T).to(DEVICE)
        solved_num_total = 0

        for states, options, solveds, lengths, masks in dataloader:
            states, options, solveds, lengths, masks = [x.to(DEVICE) for x in [states, options, solveds, lengths, masks]]
            option_logps, solved_logps = abstract.rnn_fine_tune_logps(
                    states, options, solveds, lengths, masks, control_net, rnn, params)

            B, T_plus_one, *s = states.shape
            T = T_plus_one - 1

            solved_loss = nll_loss(solved_logps, solveds)
            option_loss = nll_loss(option_logps, options)

            # ignore loss for padded elements
            option_loss = option_loss * masks[:, 1:]
            solved_loss = solved_loss * masks
            loss = option_loss.sum() + solved_loss.sum()

            train_loss += loss.item()
            # solved and option loss
            loss = loss / (2 * sum(lengths))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            updates += B

            # accuracy calculation
            option_preds = torch.argmax(option_logps, dim=1)
            solved_preds = torch.argmax(solved_logps, dim=1)
            assert_shape(option_preds, (B, T))

            # masks[:, :T+1] = 1, which is one more than the number of options.
            # to get correct mask, take masks[:, 1:]
            matched = ((option_preds == options) * masks[:, 1:]).sum(dim=0)
            solved_matched = ((solved_preds == solveds) * masks).sum(dim=0)
            option_num_correct[:len(matched)] += matched
            solved_num_correct[:len(solved_matched)] += solved_matched
            tried = masks[:, 1:].sum(dim=0)
            option_num_total[:len(tried)] += tried
            solved_num_total += masks.sum()


        for i in range(len(option_num_total)):
            if option_num_total[i] == 0:
                option_num_total[i] = 1

        acc_dict = {f'acc{i}': (0 if (option_num_total[i] == 0)
                                  else option_num_correct[i] / option_num_total[i])
                    for i in range(len(option_num_correct))}
        wandb.log({'loss': train_loss,
                   'acc': option_num_correct.sum() / option_num_total.sum(),
                   'solved_acc': solved_num_correct.sum() / solved_num_total,
                   **acc_dict})

        eval_option_num_correct = torch.zeros(dataset.max_T).to(DEVICE)
        eval_solved_num_correct = torch.zeros(dataset.max_T + 1).to(DEVICE)
        eval_option_num_total = torch.zeros(dataset.max_T).to(DEVICE)
        eval_solved_num_total = 0
        # evaluation accuracy calculation
        for states, options, solveds, lengths, masks in eval_dataloader:
            states, options, solveds, lengths, masks = [x.to(DEVICE) for x in [states, options, solveds, lengths, masks]]
            option_logps, solved_logps = abstract.rnn_fine_tune_logps(
                    states, options, solveds, lengths, masks, control_net, rnn, params)

            B, T_plus_one, *s = states.shape
            T = T_plus_one - 1

            # accuracy calculation
            option_preds = torch.argmax(option_logps, dim=1)
            solved_preds = torch.argmax(solved_logps, dim=1)
            assert_shape(option_preds, (B, T))

            # masks[:, :T+1] = 1, which is one more than the number of options.
            # to get correct mask, take masks[:, 1:]
            matched = ((option_preds == options) * masks[:, 1:]).sum(dim=0)
            solved_matched = ((solved_preds == solveds) * masks).sum(dim=0)
            eval_option_num_correct[:len(matched)] += matched
            eval_solved_num_correct[:len(solved_matched)] += solved_matched
            tried = masks[:, 1:].sum(dim=0)
            eval_option_num_total[:len(tried)] += tried
            eval_solved_num_total += masks.sum()


        for i in range(len(eval_option_num_total)):
            if eval_option_num_total[i] == 0:
                eval_option_num_total[i] = 1

        eval_acc_dict = {f'eval_acc{i}': (0 if (eval_option_num_total[i] == 0)
                                  else eval_option_num_correct[i] / eval_option_num_total[i])
                    for i in range(len(option_num_correct))}

        wandb.log({'eval_acc': eval_option_num_correct.sum() / eval_option_num_total.sum(),
                   'solved_acc': eval_solved_num_correct.sum() / eval_solved_num_total,
                   **eval_acc_dict})

        if (not params.no_log and params.save_every
                and (time.time() - last_save_time > (params.save_every * 60))):
            last_save_time = time.time()
            path = utils.save_model(control_net, f'models/{params.id}-epoch-{epoch}_control.pt')
            path2 = utils.save_model(rnn, f'models/{params.id}-epoch-{epoch}_rnn.pt')

        epoch += 1

    if not params.no_log and params.save_every:
        path = utils.save_model(control_net, f'models/{params.id}_control.pt')
        path2 = utils.save_model(rnn, f'models/{params.id}_rnn.pt')


def fine_tune(control_net, params):
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

        epoch += 1

    if not params.no_log and params.save_every:
        path = utils.save_model(control_net, f'models/{params.id}_control.pt')


def learn_options(net, params):
    env = box_world.BoxWorldEnv(seed=params.seed, solution_length=params.solution_length,
                                random_goal=params.random_goal)
    test_env = box_world.BoxWorldEnv(solution_length=params.solution_length, random_goal=params.random_goal)

    dataset = data.BoxWorldDataset(env, n=params.n, traj=True)
    # log the first 15 initial states to wandb for inspection
    wandb.log({'initial_states': [wandb.Image(box_world.to_color_obs(states[0])) for states, moves in dataset.data[:15]]})

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, collate_fn=data.traj_collate)

    params.epochs = int(params.traj_updates / params.n)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    net.train()

    if params.options_fine_tune:
        # micro net, macro policy net, tau net, transition net, solved net
        net.control_net.micro_net.requires_grad_(False)
        # net.control_net.tau_net.requires_grad_(False)
        # net.control_net.macro_transition_net.requires_grad_(False)


    num_params = utils.num_params(net)
    print(f"Net has {num_params} parameters")
    wandb.config.params = num_params

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
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)

            loss = net(s_i_batch, actions_batch, lengths, masks, abstract_pen=params.abstract_pen)

            train_loss += loss.item()
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            loss.backward()
            optimizer.step()

        wandb.log({'loss': train_loss})
        if params.gumbel:
            wandb.log({'gumbel_temp': abstract.GUMBEL_TEMP})
        if params.toy_test:
            print(f'{train_loss=}')

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
            wandb.log({'acc': test_acc})

        if (not params.no_log and params.save_every
                and (time.time() - last_save_time > (params.save_every * 60))):
            last_save_time = time.time()
            path = utils.save_model(net, f'models/{params.id}-epoch-{epoch}.pt')

        epoch += 1
        updates += params.n

    if not params.no_log:
        path = utils.save_model(net, f'models/{params.id}.pt')


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
            # if isinstance(net, ShrinkingRelationalDRLNet):
            #    loss += net.shrink_loss()

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

    obs_info = []

    while updates < params.traj_updates:
        total_hard_matches = 0
        total_negatives = 0
        total_positives = 0
        total_negative_matches = 0
        total_positive_matches = 0
        spotwise_correct = None

        train_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            if spotwise_correct is None:
                spotwise_correct = torch.zeros(targets.shape[1:], device=DEVICE)

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = net(inputs)
            # greedily convert probabilities to hard predictions
            hard_preds = torch.round(preds)
            # calculate number of hard matches by iterating through and counting perfect matches
            for inp, pred, target in zip(inputs, hard_preds, targets):
                negative_spots = (target == 0)
                positive_spots = (target == 1)
                match_tensor = (target == pred)
                spotwise_correct += match_tensor
                total_negatives += negative_spots.sum()
                total_positives += positive_spots.sum()
                total_negative_matches += (negative_spots * match_tensor).sum()
                total_positive_matches += (positive_spots * match_tensor).sum()

                assert torch.where(inp[:, 0, 0] != 0)[0] - 3 == torch.where(target[neurosym.HELD_KEY_IX, :, 0] != 0)[0]

                if torch.equal(pred, target):
                    total_hard_matches += 1
                elif epoch > 50:
                    if torch.any(pred[0, :, 0] != target[0, :, 0]):
                        obs_info.append((inp, pred[0, :, 0], target[0, :, 0]))

            loss = F.binary_cross_entropy(preds, targets, reduction='mean')
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        acc = total_hard_matches / len(dataloader.dataset)
        spotwise_acc = spotwise_correct / len(dataloader.dataset)
        negative_acc = (total_negative_matches / total_negatives).item()
        positive_acc = (total_positive_matches / total_positives).item()

        wandb.log({'loss': train_loss,
                   'acc': acc,
                   'negative_acc': negative_acc,
                   'positive_acc': positive_acc})

        if epoch % 10 == 0:
            print(f'{train_loss=}, {acc=}, {negative_acc=}, {positive_acc=}')
            print(f"{spotwise_acc=}")

            if epoch > 50:
                if len(obs_info) > 1:
                    print(f"{net.conv1.weight=}")
                    if len(obs_info) > 10:
                        random.shuffle(obs_info)
                        obs_info = obs_info[:3]
                    for obs, pred_keys, target_keys in obs_info:
                        # convert from tensor to ascii
                        ascii_obs = box_world.tensor_to_obs(obs)
                        print(f"{pred_keys=}")
                        print(f"{target_keys=}")
                        print(ascii_obs)

        epoch += 1
        updates += len(dataloader.dataset)

    if not params.no_log and params.save_every:
        path = utils.save_model(net, f'models/{params.id}_neurosym.pt')


def learn_neurosym_world_model(params):
    env = box_world.BoxWorldEnv()
    n = params.n
    path = f'data/abstract-{n}'
    if not os.path.isfile(path):
        print(f'no dataset found at {path}, generating dataset from scratch')
        dataset = data.ListDataset(neurosym.world_model_data(env, n=n))
        torch.save(dataset, path)
        print(f'saved dataset at {path}')
    else:
        print(f'loaded dataset from {path}')
        dataset = torch.load(path)

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    C = box_world.NUM_COLORS
    net = nn.Sequential(RelationalDRLNet(input_channels=box_world.NUM_ASCII, out_dim=2 * C * C * 2),
                        Rearrange('b (p C1 C2 two) -> b p C1 C2 two', p=2, C1=C, C2=C, two=2),
                        nn.LogSoftmax(dim=-1),
                        ).to(DEVICE)

    print(f"Net has {utils.num_params(net)} parameters")
    options_net = neurosym.SVOptionNet(num_colors=C,
                                       num_options=C,
                                       hidden_dim=128,
                                       num_hidden=2).to(DEVICE)

    optimizer = torch.optim.Adam(chain(net.parameters(), options_net.parameters()), lr=params.lr)

    params.epochs = int(params.traj_updates / params.n)

    updates = 0
    epoch = 0

    # eval_dataset = torch.load('data/abstract-eval-100')
    # eval_dataloader = DataLoader(eval_dataset, batch_size=params.batch_size, shuffle=True)
    # eval_env = box_world.BoxWorldEnv(seed=2)
    # eval_dataset = data.ListDataset(neurosym.world_model_data(env, n=100))
    # torch.save(eval_dataset, 'data/abstract-eval-100')

    while updates < params.traj_updates:
        total_loss = 0
        total_cc_loss = 0
        total_move_loss = 0
        total_state_loss = 0

        move_preds_num_right = 0
        total_move_preds = 0
        state_preds_num_right = 0
        total_state_preds = 0

        for states, moves, next_states, correct_state_embeds, correct_next_state_embeds in dataloader:
            optimizer.zero_grad()

            states = states.to(DEVICE)
            moves = moves.to(DEVICE)
            next_states = next_states.to(DEVICE)
            correct_state_embeds = correct_state_embeds.to(DEVICE)
            correct_next_state_embeds = correct_next_state_embeds.to(DEVICE)

            state_embeds = net(states)
            # B, p, C, C, 2
            assert not torch.any(state_embeds.isnan())

            with torch.no_grad():
                next_state_embeds = net(next_states)

            state_loss = F.kl_div(state_embeds, correct_state_embeds, log_target=True, reduction='sum')
            state_loss = state_loss / state_embeds.numel()

            move_logits = options_net(state_embeds)
            move_precond_logps = neurosym.precond_logps(state_embeds)
            move_logits = move_logits * move_precond_logps
            move_loss = F.cross_entropy(move_logits, moves, reduction='mean')

            next_state_preds = neurosym.world_model_step(state_embeds, moves, neurosym.BW_WORLD_MODEL_PROGRAM)
            cc_loss = F.kl_div(next_state_preds, next_state_embeds, log_target=True, reduction='none')
            bad_ixs = torch.where(cc_loss > 1)
            if len(bad_ixs[0]) > 0:
                print(f"{bad_ixs=}")
                print(f"{cc_loss[bad_ixs]=}")
                print(f"{state_embeds[bad_ixs]=}")
                print(f"{next_state_preds[bad_ixs]=}")
                print(f"{next_state_embeds[bad_ixs]=}")
                print(f"{state_embeds[0,0,:3, :3, 0]=}")
                print(f"{next_state_preds[0,0,:3, :3, 0]=}")
                print(f"{next_state_embeds[0,0,:3, :3, 0]=}")
                print(f"{cc_loss[0,0,:3, :3, 0]=}")
            cc_loss = cc_loss.sum()

            # manual balancing so cc loss is same order of magnitude as state loss..
            cc_loss = 10 * cc_loss / next_state_preds.numel()

            loss = torch.tensor(0., requires_grad=True)
            if params.state_loss:
                loss = loss + state_loss
            if params.move_loss:
                loss = loss + move_loss
            if params.cc_loss:
                loss = loss + cc_loss

            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_move_loss += move_loss.item()
            total_cc_loss += cc_loss.item()

            state_preds = torch.round(state_embeds.exp())
            correct_state_embeds = torch.round(correct_state_embeds.exp())
            state_preds_num_right += (state_preds == correct_state_embeds).sum()
            total_state_preds += state_preds.numel()

            move_preds = torch.argmax(move_logits, dim=1)
            move_preds_num_right += (move_preds == moves).sum()
            total_move_preds += moves.numel()

            loss.backward()
            optimizer.step()

        wandb.log({'loss': total_loss,
                   'move_loss': total_move_loss,
                   'state_loss': total_state_loss,
                   'cc_loss': total_cc_loss,
                   'move_acc': move_preds_num_right / total_move_preds,
                   'state_acc': state_preds_num_right / total_state_preds,
                   })

        epoch += 1
        updates += len(dataloader.dataset)


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
            net = CausalNet(control_net, abstract_pen=params.abstract_pen, cc_weight=params.cc_weight)
        else:
            raise NotImplementedError()

    return net


def boxworld_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--traj_updates', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--b', type=int, default=10, help='number of options')
    parser.add_argument('--abstract_pen', type=float, default=1.0, help='for starting a new option, this penalty is subtracted from the overall logp of the seq')
    parser.add_argument('--model', type=str, default='cc', choices=['sv', 'cc', 'hmm-homo', 'hmm'])
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

    parser.add_argument('--solution_length', type=tuple, default=(1, 2, 3, 4),
                        help='box world env solution_length, may be single number or tuple of options')
    parser.add_argument('--muzero', action='store_true')
    parser.add_argument('--muzero_scratch', action='store_true')
    parser.add_argument('--num_test', type=int, default=200)
    parser.add_argument('--test_every', type=float, default=60, help='number of minutes to test every, if false will not test')
    parser.add_argument('--save_every', type=float, default=180, help='number of minutes to save every, if false will not save')
    parser.add_argument('--neurosym', action='store_true')
    parser.add_argument('--cc_neurosym', action='store_true')
    parser.add_argument('--sv_options', action='store_true')
    parser.add_argument('--dim', type=int, default=64, help='latent dim of relational net')
    parser.add_argument('--num_attn_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--cc_weight', type=float, default=1.0)
    parser.add_argument('--fake_cc_neurosym', action='store_true', help='hard coded state abstraction fn')
    parser.add_argument('--symbolic_sv', action='store_true')
    parser.add_argument('--micro_net2', action='store_true')
    parser.add_argument('--num_out', type=int, default=None)
    parser.add_argument('--check_ix', type=int, default=-1)
    parser.add_argument('--num_check', type=int, default=0)
    parser.add_argument('--sv_micro', action='store_true')
    parser.add_argument('--sv_micro_data_type', type=str, default='full_traj')
    parser.add_argument('--relational_macro', action='store_true')
    parser.add_argument('-M', '--move_loss', action='store_true', help='neurosym move loss weight')
    parser.add_argument('-S', '--state_loss', action='store_true', help='neurosym state loss weight')
    parser.add_argument('-C', '--cc_loss', action='store_true', help='neurosym cc loss weight')
    parser.add_argument('--rnn_macro', action='store_true', help='use RNN macro transition function in CC options learning')
    parser.add_argument('--load_rnn', action='store_true', help='load rnn for fine_tuning')
    parser.add_argument('--random_goal', action='store_true', help='random goal color trajs')
    parser.add_argument('--options_fine_tune', action='store_true',
                        help='fine tune with options learning for policy/reward predictors')
    parser.add_argument('--bigger_micro', action='store_true')
    parser.add_argument('--model_load_path', type=str, default=None)

    params = parser.parse_args()

    if params.num_check > 0:
        neurosym.CHECK_IXS = [i+1 for i in range(params.num_check)]
    else:
        neurosym.CHECK_IXS = [params.check_ix]

    utils.gpu_check()

    if not params.seed:
        seed = random.randint(0, 2**32 - 1)
        params.seed = seed

    random.seed(params.seed)
    torch.manual_seed(params.seed)

    # the parser parses as string tuple instead of int tuple
    params.solution_length = tuple(int(l) for l in params.solution_length)

    if not hasattr(params, 'batch_size'):
        if params.relational_micro or params.gumbel or params.shrink_micro_net:
            params.batch_size = 16
        elif params.neurosym:
            params.batch_size = 128
        else:
            params.batch_size = 32
        if params.ellis:  # more memory available!
            params.batch_size *= 2
        if params.abstract_dim > 64:
            params.batch_size = max(1, int(params.batch_size / 2))

    if params.cc_neurosym:
        params.model = 'cc'
        params.b = box_world.NUM_COLORS
        params.batch_size = max(1, int(params.batch_size / 2))
    if params.fake_cc_neurosym:
        params.b = box_world.NUM_COLORS

    if params.fine_tune and not params.load:
        # print('WARNING: params.load = False, creating new model')
        params.load = True

    if params.options_fine_tune:
        params.load = True
        if params.model_load_path is None:
            params.model_load_path = 'models/8110c8302c1946a5a6838cd2430b705f.pt'

    if not hasattr(params, 'traj_updates'):
        params.traj_updates = 1E8 if (params.fine_tune or params.muzero or params.neurosym) else 1E7  # default: 1E7

    params.gumbel_sched = make_gumbel_schedule_fn(params)
    params.device = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'cpu'

    if params.toy_test:
        # params.n = 100
        params.n = 10
        params.batch_size = 5
        params.traj_updates = 500
        params.test_every = 1
        params.save_every = False
        params.no_log = True
        params.num_test = 5

    if params.muzero:
        params.load = True

    with Timing('Completed training'):
        params.id = utils.generate_uuid()
        print(f"Starting run:\n{params.id}")

        print(f'{params=}')

        wandb.init(project="abstraction",
                   mode='disabled' if params.no_log else 'online',
                   config=vars(params))

        if params.muzero:
            net = make_net(params).to(DEVICE)
            data_net = net

            if params.muzero_scratch:
                params.load = False
                net = make_net(params).to(DEVICE)

            muzero.main(net.control_net, params, data_net=data_net.control_net)
        elif params.sv_micro:
            params.load = True
            if params.model_load_path is None:
                params.model_load_path = 'models/0b31d27e41b3422aa9b51e304a04516d.pt'
            net = make_net(params).to(DEVICE).control_net
            assert isinstance(net, abstract.HeteroController)
            sv_micro_train(params, net)
        elif params.fine_tune:
            net = make_net(params).to(DEVICE)
            fine_tune_rnn(net.control_net, params)
        elif params.neurosym:
            neurosym_train(params)
        elif params.sv_options:
            sv_option_pred(params)
        else:
            net = make_net(params).to(DEVICE)
            if params.rnn_macro:
                rnn = torch.nn.GRU(input_size=params.b, hidden_size=params.abstract_dim, batch_first=True).to(DEVICE)
                net.control_net.add_rnn(rnn)
            learn_options(net, params)


def neurosym_train(params):
    # seems like I have to do this outside of the function to get it to work?
    print('uncomment this to run')
    # pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')

    solution_length = (max(params.solution_length), )
    env = box_world.BoxWorldEnv(solution_length=solution_length, num_forward=(4, ))

    if params.symbolic_sv:
        dataset = neurosym.supervised_symbolic_state_abstraction_data(env, n=params.n, num_out=params.num_out)
        abs_data = data.ListDataset(dataset)
        print(f'{len(abs_data)} examples')
        dataloader = DataLoader(abs_data, batch_size=params.batch_size, shuffle=True)

        if params.micro_net2:
            if params.num_out is None:
                net = MicroNet2(input_channels=box_world.NUM_ASCII, num_colors=box_world.NUM_COLORS).to(DEVICE)
            else:
                net = MicroNet3(input_channels=box_world.NUM_ASCII, num_colors=box_world.NUM_COLORS, num_out=params.num_out).to(DEVICE)
        else:
            C = box_world.NUM_COLORS
            out_dim = 2 * C * C
            net = RelationalDRLNet(input_channels=box_world.NUM_ASCII, out_dim=out_dim, d=128)
            net = nn.Sequential(net,
                                Rearrange('b (p C1 C2) -> b p C1 C2', p=2, C1=C, C2=C),
                                nn.Sigmoid(),
                                ).to(DEVICE)

        print(f"Net has {utils.num_params(net)} parameters")
        neurosym_symbolic_supervised_state_abstraction(dataloader, net, params)
    else:
        learn_neurosym_world_model(params)


def sv_option_pred(params):
    env = box_world.BoxWorldEnv()

    # dataset of (symbolic_tensorized_state, option) pairs
    dataset = data.ListDataset(neurosym.option_sv_data(env, n=params.n))

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    # if params.sv_options_net_fc:
    net = neurosym.SVOptionNet(num_colors=box_world.NUM_COLORS,
                               num_options=box_world.NUM_COLORS,
                               hidden_dim=128,
                               num_hidden=2).to(DEVICE)
    # net = neurosym.SVOptionNet2(num_colors=box_world.NUM_COLORS,
    #                             num_options=box_world.NUM_COLORS,
    #                             num_heads=params.num_heads,
    #                             hidden_dim=128).to(DEVICE)

    print(f"Net has {utils.num_params(net)} parameters")
    option_pred_train(dataloader, net, params)


def option_pred_train(dataloader, net, params):
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    net.train()

    updates = 0
    train_loss = 0

    while updates <= params.traj_updates:
        train_loss = 0

        num_right = 0
        for datum in dataloader:
            optimizer.zero_grad()

            datum = tuple([d.to(DEVICE) for d in datum])
            states, moves = datum
            B = states.shape[0]
            move_logits = net(states)
            assert_shape(move_logits, (B, box_world.NUM_COLORS))
            assert_shape(moves, (B, ))
            assert max(moves) <= box_world.NUM_COLORS - 1
            move_preds = torch.argmax(move_logits, dim=1)
            num_right += sum(move_preds == moves)
            loss = loss_fn(move_logits, moves)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        acc = num_right / len(dataloader.dataset)
        updates += len(dataloader.dataset)
        wandb.log({'loss': train_loss,
                   'acc': acc})


def sv_micro_train(params, control_net):
    '''
    controll net is for data generation, not training
    '''
    net = abstract.ActionsAndStopsMicroNet(a=4, b=params.b, relational=params.relational_micro).to(DEVICE)
    dataset = data.sv_micro_data(n=params.n, typ=params.sv_micro_data_type, control_net=control_net)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    # can't use cross entropy, because the action_logps are already softmaxed
    # instead, use negative log likelihood
    loss_fn = nn.NLLLoss(reduction='mean')
    net.train()

    updates = 0
    train_loss = 0

    while updates <= params.traj_updates:
        train_loss = 0

        num_right = 0
        for states, actions, options in dataloader:
            optimizer.zero_grad()
            states, actions, options = states.to(DEVICE), actions.to(DEVICE), options.to(DEVICE)
            B = states.shape[0]
            action_logps, stop_logps = net(states)
            assert_shape(action_logps, (B, params.b, 4))
            action_logps = action_logps[range(B), options, :]
            assert_shape(action_logps, (B, 4))

            loss = loss_fn(action_logps, actions)
            train_loss += loss.item()

            pred = torch.argmax(action_logps, dim=1)
            num_right += sum(pred == actions)

            loss.backward()
            optimizer.step()

        acc = num_right / len(dataloader.dataset)
        updates += len(dataloader.dataset)
        wandb.log({'loss': train_loss,
                   'acc': acc})


if __name__ == '__main__':
    boxworld_main()
