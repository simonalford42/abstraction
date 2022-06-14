from typing import Any
import numpy as np
from torch.utils.data import DataLoader
import planning
import argparse
import torch
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import random
import utils
from utils import Timing, DEVICE, assert_equal
from abstract import boxworld_controller, boxworld_homocontroller
import abstract
from hmm import CausalNet, SVNet, HmmNet
import time
import box_world
import neptune.new as neptune
import mlflow
import data
from modules import FC


def fine_tune(run, env, control_net: nn.Module, params: dict[str, Any]):
    optimizer = torch.optim.Adam(control_net.parameters(), lr=params['lr'])
    control_net.train()

    epoch = 0
    updates = 0

    if params['tau_precompute']:
        control_net.freeze_microcontroller()
    else:
        control_net.freeze_all_controllers()

    env = box_world.BoxWorldEnv(seed=params['seed'])

    dataset = data.PlanningDataset(env, control_net, n=params['n'], tau_precompute=params['tau_precompute'])

    print(f'{len(dataset)} fine-tuning examples')
    params['epochs'] = int(params['traj_updates'] / len(dataset))
    print(f"actual params['epochs']: {params['epochs']}")

    if params['num_tests'] == 0:
        params['test_every'] = False
    else:
        params['test_every'] = max(1, params['epochs'] // params['num_tests'])
        print(f"actual params['test_every']: {params['test_every']}")
    if params['num_saves'] == 0:
        params['save_every'] = False
    else:
        params['save_every'] = params['epochs'] // params['num_saves']
        print(f"actual params['save_every']: {params['save_every']}")

    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True,
                            collate_fn=data.latent_traj_collate)

    while updates < params['traj_updates']:
        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle(batch_size=params['batch_size'])

        # if params['test_every'] and epoch > 0 and epoch % params['test_every'] == 0:
        #     print('recalculating dataset')
        #     env = box_world.BoxWorldEnv(seed=params['seed'] + epoch)
        #     dataset = data.PlanningDataset(env, control_net, n=params['n'], tau_precompute=tau_precompute)
        #     dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True,
        #                             collate_fn=data.latent_traj_collate)

        train_loss = 0

        first = True
        for states, options, solveds, lengths, masks in dataloader:
            states, options, solveds, lengths, masks = [x.to(DEVICE) for x in [states, options, solveds, lengths, masks]]

            B, T, *s = states.shape

            if not params['tau_precompute']:
                states_flattened = states.reshape(B * T, *s)
                t_i_flattened = control_net.tau_net(states_flattened)
                t_i = t_i_flattened.reshape(B, T, control_net.t)
            else:
                t_i = states

            if params['test_every'] and epoch % params['test_every'] == 0 and first:
                preds = control_net.macro_transitions2(t_i[:, 0], options[:, 0])
                print('first batch avg cc loss ', (t_i[:, 1] - preds).sum() / t_i.shape[0])
                first = False

            loss = abstract.fine_tune_loss_v3(t_i, options, solveds, control_net, masks)

            train_loss += loss.item()
            loss = loss / sum(lengths)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            updates += B

        if params['test_every'] and epoch % (max(1, params['test_every'] // 5)) == 0:
            print(f"train_loss: {train_loss}")

        if params['test_every'] and epoch % params['test_every'] == 0:
            utils.warn('fixed test env seed')
            print(f'Epoch {epoch}')
            planning.eval_planner(
                control_net, box_world.BoxWorldEnv(seed=env.seed), n=params['num_test'],
                # control_net, box_world.BoxWorldEnv(seed=env.seed + 1), n=params['num_test'],
            )

        if not params['no_log'] and params['save_every'] and epoch % params['save_every'] == 0 and epoch > 0:
            model_id = params['id']
            path = utils.save_model(control_net, f'models/{model_id}-epoch-{epoch}_control.pt')
            run['models'].log(path)

        epoch += 1

    if not params['no_log']:
        model_id = params['id']
        path = utils.save_model(control_net, f'models/{model_id}_control.pt')
        run['model'] = path


def train(run, dataloader: DataLoader, net: nn.Module, params: dict[str, Any]):
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
    net.train()
    model_id = mlflow.active_run().info.run_id
    run['model_id'] = model_id
    test_env = box_world.BoxWorldEnv()
    print(f"Net has {utils.num_params(net)} parameters")

    updates = 0
    epoch = 0
    while updates < params['traj_updates']:
        if params['gumbel']:
            abstract.GUMBEL_TEMP = params['gumbel_sched'](epoch / params['epochs'])

        if params['variable_abstract_pen']:
            frac = min(1, 2 * updates / params['traj_updates'])
            net.abstract_pen = params['abstract_pen'] * frac
            if epoch % (params['test_every'] // 5) == 0:
                print(f"net.abstract_pen: {net.abstract_pen}")

        if params['freeze'] is not False and updates / params['traj_updates'] >= params['freeze']:
            # net.control_net.freeze_all_controllers()
            net.control_net.freeze_microcontroller()

        # if epoch and epoch % 100 == 0:
        #     print('reloading data')
        #     del dataloader
        #     dataset = data.BoxWorldDataset(box_world.BoxWorldEnv(seed=epoch + params['seed'] * params['epochs']),
        #                                      n=params['n'], traj=True)
        #     dataloader = DataLoader(dataset, batch_size=params['batch_size'],
        #                             shuffle=False, collate_fn=data.traj_collate)

        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle(batch_size=params['batch_size'])

        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)

            loss = net(s_i_batch, actions_batch, lengths, masks)

            train_loss += loss.item()
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            run['batch/loss'].log(loss.item())
            run['batch/avg length'].log(sum(lengths) / len(lengths))
            run['batch/mem'].log(utils.get_memory_usage())
            loss.backward()
            optimizer.step()

        if epoch % (params['test_every'] // 5) == 0:
            if params['gumbel']:
                print(f"abstract.GUMBEL_TEMP: {abstract.GUMBEL_TEMP}")
            print(f"train_loss: {train_loss}")

        if params['test_every'] and epoch % params['test_every'] == 0:
            # test_env = box_world.BoxWorldEnv(seed=params['seed'])
            # print('fixed test env')
            test_acc = data.eval_options_model(
                net.control_net, test_env, n=params['num_test'],
                run=run, epoch=epoch)
            # test_acc = planning.eval_planner(
            #     net.control_net, box_world.BoxWorldEnv(seed=params['seed']), n=params['num_test'],
            # )
            run['test/accuracy'].log(test_acc)
            print(f'Epoch {epoch}\t test acc {test_acc}')
            mlflow.log_metrics({'epoch': epoch, 'test acc': test_acc}, step=epoch)

        run['epoch'].log(epoch)
        run['loss'].log(train_loss)
        run['time'].log(time.time() - start)
        mlflow.log_metrics({'epoch': epoch,
                            'loss': train_loss,
                            'time': time.time() - start}, step=epoch)

        if not params['no_log'] and params['save_every'] and epoch % params['save_every'] == 0 and epoch > 0:
            path = utils.save_model(net, f'models/{model_id}-epoch-{epoch}.pt')
            run['models'].log(path)

        epoch += 1
        updates += params['n']

    if not params['no_log']:
        path = utils.save_model(net, f'models/{model_id}.pt')
        run['model'] = path


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

            train_loss += loss.item()
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            run[f'{epoch}batch/loss'].log(loss.item())
            run[f'{epoch}batch/avg length'].log(sum(lengths) / len(lengths))
            run[f'{epoch}batch/mem'].log(utils.get_memory_usage())
            loss.backward()
            optimizer.step()

        run['epoch'].log(epoch)
        run['loss'].log(train_loss)
        run['time'].log(time.time() - start)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")


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


def boxworld_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--traj_updates', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--b', type=int, default=10, help='number of options')
    parser.add_argument('--abstract_pen', type=float, default=1.0, help='for starting a new option, this penalty is subtracted from the overall logp of the seq')
    parser.add_argument('--model', type=str, default='cc', choices=['sv', 'cc', 'hmm-homo', 'hmm', 'ccts', 'ccts-reduced'])
    parser.add_argument('--seed', type=int, default=1, help='seed=0 chooses a random seed')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS)

    parser.add_argument('--abstract_dim', type=int, default=32)
    parser.add_argument('--tau_noise_std', type=float, default=0.0, help='STD of N(0, sigma) noise added to abstract state embedding to aid planning')
    parser.add_argument('--freeze', type=float, default=False, help='what % through training to freeze some subnets of control net')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--ellis', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--variable_abstract_pen', action='store_true')
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
    parser.add_argument('--layer_ensemble_loss_scale', type=float, default=1)
    args = parser.parse_args()

    if not args.seed:
        seed = random.randint(0, 2**32 - 1)
        args.seed = seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.neptune:
        run = neptune.init(
            project="simonalford42/abstraction", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNDljOWE3Zi1mNzc5LTQyYjEtYTdmOC1jYTM3ZThhYjUwNzYifQ==",
        )
    else:
        run = utils.NoLogRun()
    if args.no_log:
        global mlflow
        mlflow = utils.NoMlflowRun()
    else:
        mlflow.set_experiment('Boxworld 3/22')

    batch_size = 64 if args.ellis else 32
    if args.relational_micro or args.gumbel:
        batch_size = 32 if args.ellis else 16

    if args.fine_tune and not args.load:
        print('Set load=True for fine tuning')
        args.load = True
    params = dict(
        # n=5, traj_updates=30, num_test=5, num_tests=2, num_saves=0,
        n=20000,
        traj_updates=1E9 if args.fine_tune else 1E7,  # default: 1E7
        num_saves=20, num_tests=100, num_test=200,
        lr=8E-4, batch_size=batch_size,
        # model_load_path='models/e14b78d01cc548239ffd57286e59e819.pt',
        model_load_path='models/4f33c4fd2210434ab368a39eb335d2d8-epoch-625.pt',
    )
    params.update(vars(args))
    if args.toy_test:
        params.update(dict(n=100, traj_updates=5000, num_test=5, num_tests=2, num_saves=0, no_log=True))
    featured_params = ['model', 'n', 'abstract_pen']

    # assert_equal('model_load_path' in params, params['load'])
    if params['load']:
        net = utils.load_model(params['model_load_path'])
    elif args.model == 'sv':
        net = SVNet(boxworld_homocontroller(b=1))
    else:
        if args.model == 'hmm-homo':
            # HMM algorithm where start probs, stop probs, action probs all come from single NN
            control_net = boxworld_homocontroller(b=params['b'])
        else:
            typ = 'hetero' if args.model in ['hmm', 'cc'] else args.model
            control_net = boxworld_controller(typ, params)
        if args.model in ['hmm', 'hmm-homo', 'ccts']:
            net = HmmNet(control_net, abstract_pen=params['abstract_pen'], shrink_micro_net=params['shrink_micro_net'])
        elif args.model == 'cc':
            assert not params['shrink_micro_net']
            net = CausalNet(control_net, abstract_pen=params['abstract_pen'])
        else:
            raise NotImplementedError()

    if params['gumbel']:
        plateau_percent = 0.8
        # r is set so that np.exp(-r * plateau_percent) = 0.5
        r = -np.log(0.5) / plateau_percent

        def schedule_temp(percent_through):
            # between 0.5 and 1
            x = np.exp(-r * percent_through)
            # between 0 and 1
            x = 2 * x - 1
            # between stop and start
            x = params['g_stop_temp'] + x * (params['g_start_temp'] - params['g_stop_temp'])
            return max(params['g_stop_temp'], x)
        params['gumbel_sched'] = schedule_temp

    params['device'] = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'cpu'
    params['epochs'] = int(params['traj_updates'] / params['n'])
    if params['num_tests'] == 0:
        params['test_every'] = False
    else:
        params['test_every'] = max(1, params['epochs'] // params['num_tests'])
    if params['num_saves'] == 0:
        params['save_every'] = False
    else:
        params['save_every'] = params['epochs'] // params['num_saves']

    if args.eval:
        data.eval_options_model(net.control_net, box_world.BoxWorldEnv(), n=100, option='verbose')
    elif args.fine_tune:
        net = net.to(DEVICE)
        # dataset = box_world.BoxWorldDataset(box_world.BoxWorldEnv(seed=args.seed), n=params['n'], traj=True)
        # dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=data.traj_collate)

        with Timing('Completed fine tuning'):
            with mlflow.start_run():
                params['id'] = mlflow.active_run().info.run_id
                run['params'] = params
                mlflow.log_params(params)
                for p in featured_params:
                    print(p.upper() + ':\t ' + str(params[p]))
                print(f"Starting run:\n{mlflow.active_run().info.run_id}")
                print(f"params: {params}")
                fine_tune(run, box_world.BoxWorldEnv(), net.control_net, params)

    else:
        net = net.to(DEVICE)
        dataset = data.BoxWorldDataset(box_world.BoxWorldEnv(seed=args.seed), n=params['n'], traj=True)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=data.traj_collate)

        with Timing('Completed training'):
            with mlflow.start_run():
                params['id'] = mlflow.active_run().info.run_id
                run['params'] = params
                mlflow.log_params(params)
                for p in featured_params:
                    print(p.upper() + ':\t ' + str(params[p]))
                print(f"Starting run:\n{mlflow.active_run().info.run_id}")
                print(f"params: {params}")

                train(run, dataloader, net, params)

        for p in featured_params:
            print(p.upper() + ': \t' + str(params[p]))

    run.stop()


if __name__ == '__main__':
    boxworld_main()
