from typing import Any
import numpy as np
from modules import FC
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

    for epoch in range(params['epochs']):
        if hasattr(dataloader.dataset, 'shuffle'):
            dataloader.dataset.shuffle()

        train_loss = 0
        start = time.time()
        with Timing('Completed epoch'):
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
                test_acc = box_world.eval_options_model(net.control_net, test_env, n=params['num_test'], run=run, epoch=epoch)
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

        if params['save_every'] and epoch % params['save_every'] == 0:
            path = utils.save_model(net, f'models/{model_id}-epoch-{epoch}.pt')
            run['models'].log(path)
            mlflow.log_metrics({'epoch': epoch, f'model epoch {epoch}': path})

    path = utils.save_model(net, f'models/{model_id}.pt')
    run['model'] = path
    mlflow.log_params({f'final model path': path})


def sv_train(run, dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
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

        metrics = dict(
            epoch=epoch,
            loss=train_loss,
        )

        run['epoch'].log(epoch)
        run['loss'].log(train_loss)
        run['time'].log(time.time() - start)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")


def boxworld_outer_sv(run, dataloader, net, params, epochs=100, rounds=-1):
    print('outer sv')
    env = box_world.BoxWorldEnv()
    # print_every = epochs / 5
    print_every = 10
    save_every = 1
    print(f"Net has {utils.num_params(net)} parameters")
    model_id = mlflow.active_run().info.run_id
    run['model_id'] = model_id

    try:
        round = 0
        test_acc = 0
        while round != rounds:
            start = time.time()
            print(f'Round {round}')

            with Timing("Generated trajectories"):
                dataloader = box_world.box_world_dataloader(env=env, n=params['n'], traj=True, batch_size=params['batch_size'])

            sv_train(run, dataloader, net, epochs=epochs, lr=params['lr'], print_every=print_every)
            mlflow.log_metrics({'epoch': round,
                                'loss': round,
                                'time': time.time() - start}, step=round)

            if test_every and round % test_every == 0:
                with Timing("Evaluated model"):
                    if net.b != 1:
                        test_acc = box_world.eval_options_model(net.control_net, env, n=params['num_test'])
                    else:
                        test_acc = box_world.eval_model(net.control_net, env, n=params['num_test'])
                    run['test/accuracy'].log(test_acc)
                    print(f'Epoch {epoch}\t test acc {test_acc}')
                    mlflow.log_metrics({'epoch': round, 'test acc': test_acc}, step=round)
            else:
                mlflow.log_metrics({'epoch': round, 'test acc': test_acc}, step=round)

            # if save_every and round % save_every == 0:
                # utils.save_mlflow_model(net, model_name=f'round-{round}', overwrite=False)

            round += 1
    except KeyboardInterrupt:
        pass

    path = utils.save_model(net, f'models/{model_id}.pt')
    run['model'] = path
    mlflow.log_params({f'final model path': path})


def up_right_main():
    scale = 3
    seq_len = 5
    trajs = up_right.generate_data(scale, seq_len, n=100)

    data = up_right.TrajData(trajs)

    s = data.state_dim
    abstract_policy_net = HeteroController(
        a := 2, b := 2, t := 10,
        tau_net=FC(s, t, hidden_dim=64, num_hidden=1),
        micro_net=FC(s, b * a, hidden_dim=64, num_hidden=1),
        stop_net=FC(s, b * 2, hidden_dim=64, num_hidden=1),
        start_net=FC(t, b, hidden_dim=64, num_hidden=1),
        alpha_net=FC(t + b, t, hidden_dim=64, num_hidden=1))
    # net = Eq2Net(abstract_policy_net,
    net = HmmNet(abstract_policy_net)

    # utils.load_model(net, f'models/model_1-21__4.pt')
    sv_train(data, net, epochs=100, lr=1E-4)
    # utils.save_model(net, f'models/model_9-17.pt')
    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)

    viterbi(net, eval_data,)


def eval_models():
    run_id = None
    utils.load_mlflow_model(net, run_id, model_name)

    env = box_world.BoxWorldEnv(seed=1)
    box_world.eval_options_model(net.control_net, env, n=200, option='verbose')


def boxworld_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', type=float, default=1.0)
    parser.add_argument('--abstract_pen', type=float, default=0.0)
    parser.add_argument('--hmm', action='store_true')
    parser.add_argument('--sv', action='store_true')
    parser.add_argument('--disk', action='store_true')
    parser.add_argument('--homo', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--outer', action='store_true')
    args = parser.parse_args()

    random.seed(1)
    torch.manual_seed(1)

    if args.neptune:
        run = neptune.init(
            project="simonalford42/abstraction",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNDljOWE3Zi1mNzc5LTQyYjEtYTdmOC1jYTM3ZThhYjUwNzYifQ==",
        )
    else:
        run = utils.NoLogRun()
    if args.no_log:
        global mlflow
        mlflow = utils.NoMlflowRun()
    else:
        mlflow.set_experiment('Boxworld 3/22')

    params = dict(
        n=1000,
        lr=8E-4, epochs=1, batch_size=10, b=10,
        cc_weight=args.cc, abstract_pen=args.abstract_pen,
        hmm=args.hmm, homo=args.homo, sv=args.sv,
        save_every=False, test_every=10, num_test=1,
        no_log=args.no_log,
        # model_load_path='models/6956b627.pt',
        disk_data=args.disk,
        outer=args.outer,
    )

    if 'model_load_path' in params:
        net = utils.load_model(params['model_load_path'])
    else:
        if args.homo:
            assert args.hmm
            control_net = boxworld_homocontroller(b=params['b'])
        else:
            control_net = boxworld_controller(b=params['b'])
        if args.hmm:
            net = HmmNet(control_net, abstract_pen=params['abstract_pen'])
            model_type = 'hmm'
        if args.sv:
            net = SVNet(boxworld_homocontroller(b=1))
            model_type = 'sv'
        else:
            net = CausalNet(control_net, cc_weight=params['cc_weight'], abstract_pen=params['abstract_pen'])
            model_type = 'causal'
    params['model_type'] = model_type
    params['device'] = torch.cuda.get_device_name(DEVICE)

    net = net.to(DEVICE)
    if params['disk_data']:
        data = 'default' + str(int(params['n']/1000)) + 'k'
        params['data'] = data
        data = box_world.DiskData(name=params['data'], n=params['n'])
    else:
        data = box_world.BoxWorldDataset(box_world.BoxWorldEnv(), n=params['n'], traj=True)
    dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=False, collate_fn=box_world.traj_collate)

    with Timing('Completed training'):
        with mlflow.start_run():
            params['id'] = mlflow.active_run().info.run_id
            run['params'] = params
            mlflow.log_params(params)
            print(f"Starting run:\n{mlflow.active_run().info.run_id}")
            print(f"params: {params}")

            if params['outer']:
                rounds = params['epochs'] // params['test_every']
                epochs = params['epochs'] // rounds
                boxworld_outer_sv(run, dataloader, net, params, epochs=epochs, rounds=rounds)
            else:
                train(run, dataloader, net, params)

    run.stop()


if __name__ == '__main__':
    # boxworld_main()
    model_load_path = 'models/d1b71848613045649b9f9e3dd788978f.pt'
    net = utils.load_model(model_load_path)
    # env = box_world.BoxWorldEnv(max_num_steps=70)
    env = box_world.BoxWorldEnv(max_num_steps=70, solution_length=(4, ), num_forward=(1, 2, 3, 4), branch_length=2)
    box_world.eval_options_model(net.control_net, env, n=50, option='verbose')
