from typing import Any
import numpy as np
import up_right
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import random
import utils
from utils import Timing, DEVICE
from abstract import HeteroController, boxworld_controller, boxworld_homocontroller
from hmm import CausalNet, SVNet, HmmNet, viterbi
import time
import box_world
import neptune.new as neptune
import mlflow


def train(run, dataloader: DataLoader, net: nn.Module, params: dict[str, Any]):
    print('train')
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
    net.train()
    model_id = mlflow.active_run().info.run_id
    run['model_id'] = model_id
    test_env = box_world.BoxWorldEnv()
    print(f"Net has {utils.num_params(net)} parameters")

    updates = 0
    epoch = 0
    while updates < params['traj_updates']:
        if params['freeze'] is not False and updates / params['traj_updates'] >= params['freeze']:
            net.control_net.freeze_all_controllers()

        if epoch and epoch % 100 == 0:
            print('reloading data')
            del dataloader
            data = box_world.BoxWorldDataset(box_world.BoxWorldEnv(seed=epoch + params['seed'] * params['epochs']),
                                             n=params['n'], traj=True)
            dataloader = DataLoader(data, batch_size=params['batch_size'],
                                    shuffle=False, collate_fn=box_world.traj_collate)

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

        if params['test_every'] and epoch % params['test_every'] == 0:
            if net.b != 1:
                test_acc = box_world.eval_options_model(
                    net.control_net, test_env, n=params['num_test'],
                    run=run, epoch=epoch)
            else:
                test_acc = box_world.eval_model(net.control_net, test_env, n=params['num_test'])
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
    parser.add_argument('--cc_pen', type=float, default=1.0, help='causal consistency loss weight')
    parser.add_argument('--abstract_pen', type=float, default=1.0, help='for starting a new option, this penalty is subtracted from the overall logp of the seq')
    parser.add_argument('--model', type=str, default='cc', choices=['sv', 'cc', 'hmm-homo', 'hmm', 'ccts', 'ccts-reduced'])
    parser.add_argument('--seed', type=int, default=1, help='seed=0 chooses a random seed')
    parser.add_argument('--tau_noise', type=float, default=0.0, help='STD of N(0, sigma) noise added to abstract state embedding to aid planning')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--n', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--abstract_dim', type=int, default=32)
    parser.add_argument('--ellis', action='store_true')
    parser.add_argument('--freeze', type=float, default=False, help='what % through training to freeze some subnets of control net')
    parser.add_argument('--mlp_hidden_dim', type=int, default=64)
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
    params = dict(
        # n=5, traj_updates=30, num_test=5, num_tests=2, num_saves=0,
        n=5000,
        traj_updates=1E7,  # default: 1E7
        num_saves=4, num_tests=20, num_test=200,
        lr=8E-4, batch_size=batch_size, b=10,
        model_load_path='models/e14b78d01cc548239ffd57286e59e819.pt',
    )
    params.update(vars(args))
    featured_params = ['model', 'abstract_pen', 'tau_noise']

    if 'model_load_path' in params:
        net = utils.load_model(params['model_load_path'])
    elif args.model == 'sv':
        net = SVNet(boxworld_homocontroller(b=1))
    else:
        if args.model == 'hmm-homo':
            # HMM algorithm where start probs, stop probs, action probs all come from single NN
            control_net = boxworld_homocontroller(b=params['b'])
        else:
            typ = 'hetero' if args.model in ['hmm', 'cc'] else args.model
            control_net = boxworld_controller(b=params['b'], typ=typ, tau_noise_std=args.tau_noise,
                                              t=params['abstract_dim'], num_hidden=params['num_hidden'], hidden_dim=params['hidden_dim'])
        if args.model in ['hmm', 'hmm-homo', 'ccts']:
            net = HmmNet(control_net, abstract_pen=params['abstract_pen'], ccts=(args.model == 'ccts'))
        elif args.model == 'cc':
            net = CausalNet(control_net, cc_weight=params['cc_pen'], abstract_pen=params['abstract_pen'])
        else:
            raise NotImplementedError()

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
        box_world.eval_options_model(net.control_net, box_world.BoxWorldEnv(), n=100, option='verbose')
    else:
        net = net.to(DEVICE)
        data = box_world.BoxWorldDataset(box_world.BoxWorldEnv(seed=args.seed), n=params['n'], traj=True)
        dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=False, collate_fn=box_world.traj_collate)

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
