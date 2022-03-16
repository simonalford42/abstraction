from typing import Any
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


def train(run, dataloader: DataLoader, net: nn.Module, params: dict[str, Any]):
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
    net.train()

    for epoch in range(params['epochs']):
        print(f'epoch: {epoch}')
        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            print('batch')
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)
            loss = net(s_i_batch, actions_batch, lengths, masks)

            train_loss += loss
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            loss.backward()
            optimizer.step()

        if params['test_every'] and epoch % params['test_every'] == 0:
            env = box_world.BoxWorldEnv(seed=epoch)
            test_acc = box_world.eval_options_model(net.control_net, env, n=params['num_test'])
            run["test/accuracy"].log(test_acc)

        # metrics = dict(
        #     epoch=epoch,
        #     loss=loss.item(),
        # )
        # tb.add_scalars('main_tag', metrics)

        if params['print_every'] and epoch % params['print_every'] == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")
        if params['save_every'] and epoch % params['save_every'] == 0:
            utils.save_model(net, 'models/temp_save.pt')


def sv_train(tb, dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch, masks = s_i_batch.to(DEVICE), actions_batch.to(DEVICE), masks.to(DEVICE)
            loss = net(s_i_batch, actions_batch, lengths, masks)

            train_loss += loss
            # reduce just like cross entropy so batch size doesn't affect LR
            loss = loss / sum(lengths)
            loss.backward()
            optimizer.step()

        metrics = dict(
            epoch=epoch,
            loss=loss.item(),
        )
        # tb.add_scalars('main_tag', metrics)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")
        # if save_every and epoch % save_every == 0:
            # utils.save_mlflow_model(net, model_name=f"epoch-{epoch}")


def boxworld_outer_sv(
    net, n=1000, epochs=100, rounds=-1, num_test=100, test_every=1, lr=1E-4,
    batch_size=10, fix_seed: bool = False,
):
    with SummaryWriter() as tb:
        env = box_world.BoxWorldEnv()
        # print_every = epochs / 5
        print_every = 1
        save_every = 1
        params = dict(epochs=epochs, lr=lr, n=n, batch_size=batch_size)
        print(f"params: {params}")
        # tb.add_hparams(params)
        print(f"Net has {utils.num_params(net)} parameters")

        try:
            round = 0
            while round != rounds:
                print(f'Round {round}')
                if fix_seed:
                    env = box_world.BoxWorldEnv(seed=round)

                with Timing("Generated trajectories"):
                    dataloader = box_world.box_world_dataloader(env=env, n=n, traj=True, batch_size=batch_size)

                sv_train(tb, dataloader, net, epochs=epochs, lr=lr, print_every=print_every)

                if test_every and round % test_every == 0:
                    if fix_seed:
                        env = box_world.BoxWorldEnv(seed=round)
                        print('fixed seed so eval trajs = train trajs')
                    with Timing("Evaluated model"):
                        if net.b != 1:
                            box_world.eval_options_model(net.control_net, env, n=num_test)
                        else:
                            box_world.eval_model(net, env, n=num_test)

                if save_every and round % save_every == 0:
                    utils.save_mlflow_model(net, model_name=f'round-{round}', overwrite=False)

                round += 1
        except KeyboardInterrupt:
            utils.save_mlflow_model(net, overwrite=False)


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

    run = neptune.init(
        project="simonalford42/abstraction",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNDljOWE3Zi1mNzc5LTQyYjEtYTdmOC1jYTM3ZThhYjUwNzYifQ==",
    )
    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', type=float, default=1.0)
    parser.add_argument('--abstract_pen', type=float, default=0.0)
    parser.add_argument('--hmm', action='store_true')
    parser.add_argument('--homo', action='store_true')
    args = parser.parse_args()

    params = dict(
        lr=8E-4, num_test=10, epochs=20, b=10, batch_size=5,
        cc_weight=args.cc, abstract_pen=args.abstract_pen,
        hmm=args.hmm, homo=args.homo,
        data='default10', n=10,
        print_every=1, save_every=1, test_every=5,
    )

    if args.homo:
        assert args.hmm
        control_net = boxworld_homocontroller(b=params['b'])
    else:
        control_net = boxworld_controller(b=params['b'])

    if args.hmm:
        net = HmmNet(control_net, abstract_pen=params['abstract_pen']).to(DEVICE)
        # net = SVNet(homo_controller).to(DEVICE)
    else:
        net = CausalNet(control_net, cc_weight=params['cc_weight'], abstract_pen=params['abstract_pen']).to(DEVICE)

    data = box_world.DiskData(name=params['data'], n=params['n'])
    dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=False, collate_fn=box_world.traj_collate)
    with Timing('Completed training'):
        train(run, dataloader, net, params)

    run.stop()


if __name__ == '__main__':
    boxworld_main()
    # box_world.generate_data(box_world.BoxWorldEnv(), 'default10', n=10, overwrite=True)
