import up_right
import torch
from torch.utils.data import DataLoader
import random
import utils
from utils import Timing, DEVICE
import abstract
from abstract2 import Controller, TrajNet, HMMTrajNet
import time
from modules import FC, RelationalDRLNet, abstract_out_dim
import box_world
import mlflow
import hmm


def train_abstractions(dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        # total = 0
        # total_correct = 0
        for s_i_batch, actions_batch, lengths in dataloader:
            optimizer.zero_grad()
            s_i_batch = s_i_batch.to(DEVICE)
            actions_batch = actions_batch.to(DEVICE)

            # loss, correct = net(s_i_batch, actions_batch, lengths)
            loss = net(s_i_batch, actions_batch, lengths)

            # total += sum(lengths)
            # total_correct += correct

            # want total loss here
            train_loss += loss
            # need to reduce by mean, just like cross entropy, so batch size
            # doesn't affect LR.
            loss = loss / sum(lengths)

            loss.backward()
            optimizer.step()

        # acc = (total_correct / total).item()
        metrics = dict(
            epoch=epoch,
            loss=loss.item(),
            # acc=acc,
        )
        mlflow.log_metrics(metrics, step=epoch)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  # + f"acc: {acc:.3f}\t"
                  + f"({time.time() - start:.1f}s)")
        if save_every and epoch % save_every == 0:
            utils.save_mlflow_model(net, model_name=f"epoch-{epoch}")


def traj_box_world_sv_train(
    net, n=1000, epochs=100, rounds=-1, num_test=100, test_every=1, lr=1E-4,
    batch_size=10, fix_seed: bool = False,
):
    mlflow.set_experiment("Boxworld traj sv train")
    with mlflow.start_run():
        env = box_world.BoxWorldEnv()
        print_every = epochs / 5
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

                train_abstractions(dataloader, net, epochs=epochs, lr=lr,
                                   print_every=print_every)

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
    abstract_policy_net = abstract.AbstractPolicyNet(
        a := 2, b := 2, t := 10,
        tau_net=FC(s, t, hidden_dim=64, num_hidden=1),
        micro_net=FC(s, b * a, hidden_dim=64, num_hidden=1),
        stop_net=FC(s, b * 2, hidden_dim=64, num_hidden=1),
        start_net=FC(t, b, hidden_dim=64, num_hidden=1),
        alpha_net=FC(t + b, t, hidden_dim=64, num_hidden=1))
    # net = Eq2Net(abstract_policy_net,
    net = HMMTrajNet(abstract_policy_net)

    # utils.load_model(net, f'models/model_1-21__4.pt')
    train_abstractions(data, net, epochs=100, lr=1E-4)
    # utils.save_model(net, f'models/model_9-17.pt')
    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)

    hmm.eval_viterbi(net, eval_data,)
    # abstract.sample_trajectories(net, eval_data, full_abstract=False)


def traj_box_world_batched_main():
    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    # standard: n = 5000, epochs = 100, num_test = 200, lr = 8E-4, rounds = 10
    hmm = True
    n = 5000
    epochs = 100
    num_test = 200
    lr = 8E-4
    rounds = 20
    fix_seed = False
    b = 10
    batch_size = 10

    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                      num_attn_blocks=2,
                                      num_heads=4,
                                      out_dim=abstract_out_dim(a=4, b=b)).to(DEVICE)
    control_net = Controller(
        a=4,
        b=b,
        net=relational_net,
        batched=True,
    )

    if hmm:
        print('hmm training!')
        print(f"b: {b}")
        net = HMMTrajNet(control_net).to(DEVICE)
        utils.load_mlflow_model(net, run_id='d66d14463f2041d6928f93f48ee26cc6', model_name='round-19')
    else:
        net = TrajNet(control_net).to(DEVICE)

    env = box_world.BoxWorldEnv(seed=1)

    box_world.eval_options_model(net.control_net, env, n=200)
    traj_box_world_sv_train(
        net, n=n, epochs=epochs, num_test=num_test, test_every=1, rounds=rounds,
        lr=lr, batch_size=batch_size, fix_seed=fix_seed)


if __name__ == '__main__':
    traj_box_world_batched_main()
