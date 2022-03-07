from modules import FC
import up_right
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import utils
from utils import Timing, DEVICE
from abstract import HeteroController, boxworld_controller, boxworld_homocontroller
from hmm import CausalNet, TrajNet, HmmNet, viterbi
import time
import box_world
import mlflow


def sv_train(dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        for s_i_batch, actions_batch, lengths, masks in dataloader:
            optimizer.zero_grad()
            s_i_batch, actions_batch = s_i_batch.to(DEVICE), actions_batch.to(DEVICE)
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
        mlflow.log_metrics(metrics, step=epoch)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"({time.time() - start:.1f}s)")
        if save_every and epoch % save_every == 0:
            utils.save_mlflow_model(net, model_name=f"epoch-{epoch}")


def boxworld_outer_sv(
    net, n=1000, epochs=100, rounds=-1, num_test=100, test_every=1, lr=1E-4,
    batch_size=10, fix_seed: bool = False,
):
    mlflow.set_experiment("Boxworld sv train")
    with mlflow.start_run():
        env = box_world.BoxWorldEnv()
        # print_every = epochs / 5
        print_every = 1
        save_every = 1
        params = dict(epochs=epochs, lr=lr, n=n)
        print(f"params: {params}")
        mlflow.log_params(params)
        print(f"Net has {utils.num_params(net)} parameters")

        try:
            round = 0
            while round != rounds:
                print(f'Round {round}')
                if fix_seed:
                    env = box_world.BoxWorldEnv(seed=round)

                with Timing("Generated trajectories"):
                    dataloader = box_world.box_world_dataloader(env=env, n=n, traj=True, batch_size=batch_size)

                sv_train(dataloader, net, epochs=epochs, lr=lr, print_every=print_every)

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


def boxworld_main():
    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', type=float, default=1)
    args = parser.parse_args()

    # standard: n = 5000, epochs = 100, num_test = 200, lr = 8E-4, rounds = 10
    n = 5000
    epochs = 100
    num_test = 200
    lr = 8E-4
    rounds = 20
    fix_seed = False
    b = 10
    batch_size = 10
    net = 'causal'
    cc_weight = args.cc

    print('homo controller')
    print(f"net: {net}")
    print(f"cc_weight: {cc_weight}")

    if net == 'causal':
        control_net = boxworld_controller(b=b)
        net = CausalNet(control_net).to(DEVICE)
    else:
        homo_controller = boxworld_homocontroller(b=b)
        if net == 'hmm':
            net = HmmNet(homo_controller).to(DEVICE)
        else:
            net = TrajNet(homo_controller).to(DEVICE)

    boxworld_outer_sv(
        net, n=n, epochs=epochs, num_test=num_test, test_every=1, rounds=rounds,
        lr=lr, batch_size=batch_size, fix_seed=fix_seed)

    # env = box_world.BoxWorldEnv(seed=1)
    # box_world.eval_options_model(net.control_net, env, n=200)


if __name__ == '__main__':
    boxworld_main()
