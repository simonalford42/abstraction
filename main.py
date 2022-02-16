import up_right
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import utils
from utils import Timing, DEVICE
import abstract
from abstract2 import UnbatchedTrajNet, Controller, TrajNet, HMMTrajNet
import time
from modules import FC, RelationalDRLNet, abstract_out_dim
import box_world
import abstract2
import mlflow


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


def train_supervised(dataloader: DataLoader, net, epochs, lr=1E-4, save_every=None, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    net.train()

    for epoch in range(epochs):
        train_loss = 0
        start = time.time()
        correct = 0
        total = 0
        for states, actions in dataloader:
            optimizer.zero_grad()
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            pred = net(states)
            loss = criterion(pred, actions)
            train_loss += loss * states.shape[0]
            loss.backward()
            optimizer.step()

            pred2 = torch.argmax(pred, dim=1)
            correct += sum(pred2 == actions).item()
            total += len(actions)

        acc = correct / total
        metrics = dict(
            epoch=epoch,
            loss=loss.item(),
            acc=acc,
        )
        mlflow.log_metrics(metrics, step=epoch)

        if print_every and epoch % print_every == 0:
            print(f"epoch: {epoch}\t"
                  + f"train loss: {train_loss}\t"
                  + f"acc: {acc:.3f}\t"
                  + f"({time.time() - start:.1f}s)")
        if save_every and epoch % save_every == 0:
            utils.save_mlflow_model(net, model_name=f"epoch-{epoch}")


def box_world_sv_train(n=1000, epochs=100, rounds=-1, num_test=100, test_every=1, lr=1E-4):
    mlflow.set_experiment("Boxworld sv train")
    with mlflow.start_run():
        env = box_world.BoxWorldEnv()

        model_load_run_id = None
        print_every = epochs / 5
        save_every = 1

        params = dict(model_load_run_id=model_load_run_id,
                      n=n,
                      epochs=epochs,
                      lr=lr,
                      )
        print(f"params: {params}")
        mlflow.log_params(params)
        net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                               num_attn_blocks=2,
                               num_heads=4,
                               out_dim=4,
                               ).to(DEVICE)
        if model_load_run_id is not None:
            utils.load_mlflow_model(net, model_load_run_id)
        print(f"Net has {utils.num_params(net)} parameters")

        try:
            round = 0
            while round != rounds:
                print(f'Round {round}')

                with Timing("Generated trajectories"):
                    data = box_world.BoxWorldDataset(env=env, n=n, traj=False)

                print(f'{len(data)} examples')
                dataloader = DataLoader(data, batch_size=256, shuffle=True)

                train_supervised(dataloader, net, epochs=epochs, print_every=print_every, lr=lr)

                if test_every and round % test_every == 0:
                    with Timing("Evaluated model"):
                        box_world.eval_model(net, env, n=num_test)
                if save_every and round % save_every == 0:
                    utils.save_mlflow_model(net, overwrite=True)

                round += 1
                epochs = max(50, epochs - 50)
        except KeyboardInterrupt:
            utils.save_mlflow_model(net, overwrite=True)


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

    eval_viterbi(net, eval_data,)
    # abstract.sample_trajectories(net, eval_data, full_abstract=False)


def eval_viterbi(net: HMMTrajNet, data: up_right.TrajData):
    for i, s_i, actions, points in zip(range(len(data.traj_states)), data.traj_states, data.traj_moves, data.points):
        (x, y, x_goal, y_goal) = points[0][0]
        moves = ''.join(data.trajs[i])
        print(f'{moves}')
        path = abstract2.viterbi(net, s_i, actions)
        print(''.join(map(str, path)))
        print('-'*10)


def box_world_main():
    # parser = argparse.ArgumentParser(description='Abstraction')
    # parser.add_argument("--cnn",
    #                     action="store_true",
    #                     dest="cnn")
    # parser.add_argument("--test",
    #                     action="store_true",
    #                     dest="test")
    # args = parser.parse_args()
    # print(f"args: {args}")

    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    n = 5000
    epochs = 500
    num_test = min(n, 100)
    test_every = 1
    lr = 1E-4

    # net = RelationalDRLNet(input_channels=box_world.NUM_ASCII).to(DEVICE)
    # utils.load_mlflow_model(net, "1537451d1ed84d089453e238d5d92011")
    # box_world.eval_model(net, box_world.BoxWorldEnv(),
    #                      renderer=lambda obs: box_world.render_obs(obs, color=True, pause=0.001))

    box_world_sv_train(n=n, epochs=epochs, num_test=num_test, test_every=test_every, rounds=-1, lr=lr)


def batched_comparison2():
    random.seed(1)
    torch.manual_seed(2)

    a = 4
    b = 10
    t = 50
    env = box_world.BoxWorldEnv()
    data = box_world.BoxWorldDataset(env, n=2, traj=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=box_world.traj_collate)

    apn = abstract.attention_apn(b, t)
    t1, a1, s1, st1, cc1 = [], [], [], [], []

    for s_i_batch, actions_batch, lengths in dataloader:
        t_i, a, s, st, cc = apn(s_i_batch[0])
        print(f't_i: {t_i.shape}')
        t1.append(t_i)
        a1.append(a)
        s1.append(s)
        st1.append(st)
        cc1.append(cc)

    t1 = torch.cat(t1)
    a1 = torch.cat(a1)
    s1 = torch.cat(s1)
    st1 = torch.cat(st1)
    # cc1 = torch.cat(cc1)

    dataloader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=box_world.traj_collate)
    t2, a2, s2, st2, cc2 = [], [], [], [], []

    for s_i_batch, actions_batch, lengths in dataloader:
        t_i, a, s, st, cc = apn.forward_batched(s_i_batch)
        for i, max_T in enumerate(lengths):
            print(max_T)
            t2.append(t_i[i, :max_T+1])
            a2.append(a[i, :max_T+1])
            s2.append(s[i, :max_T+1])
            st2.append(st[i, :max_T+1])
            cc2.append(cc[i, :max_T+1, :max_T+1])

    t2 = torch.cat(t2)
    a2 = torch.cat(a2)
    s2 = torch.cat(s2)
    st2 = torch.cat(st2)

    torch.testing.assert_allclose(t1, t2)
    torch.testing.assert_allclose(a1, a2)
    torch.testing.assert_allclose(s1, s2)
    torch.testing.assert_allclose(st1, st2)
    for c1, c2 in zip(cc1, cc2):
        torch.testing.assert_allclose(c1, c2)
    print('all good')



def batched_comparison():
    random.seed(0)
    torch.manual_seed(0)

    a = 4
    b = 1
    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                      num_attn_blocks=4,
                                      num_heads=4,
                                      out_dim=a * b + 2 * b + b).to(DEVICE)

    control_net = Controller(
        a=4,
        b=1,
        net=relational_net,
        batched=False,
    )
    unbatched_traj_net = UnbatchedTrajNet(control_net)

    control_net = Controller(
        a=4,
        b=1,
        net=relational_net,
        batched=True,
    )
    traj_net = TrajNet(control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env, n=3, traj=True, batch_size=2)
    data = box_world.BoxWorldDataset(env, n=3, traj=True)
    dataloader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=box_world.traj_collate)

    total = 0
    for d in dataloader:
        negative_logp = traj_net(*d)
        total += negative_logp

    print(f'total0: {total}')

    total2 = 0
    for s_i, actions in zip(data.traj_states, data.traj_moves):
        negative_logp = unbatched_traj_net(s_i, actions)
        total2 += negative_logp

    print(f'total2: {total2}')

    data.traj = False
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total3 = 0
    for s_i, actions in dataloader:
        pred = relational_net(s_i)
        loss = criterion(pred[:, :4], actions)
        total3 += loss
    print(f'total3: {total3}')

    total4 = 0
    for s_i, actions in dataloader:
        pred = relational_net(s_i)
        logps = torch.log_softmax(pred[:, :4], dim=1)
        loss = -torch.sum(logps[range(len(actions)), actions])
        total4 += loss
    print(f'total4: {total4}')

    # abstract.train_supervised(dataloader, relational_net, epochs=1)


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
    # traj_box_world_sv_train(
    #     net, n=n, epochs=epochs, num_test=num_test, test_every=1, rounds=rounds,
    #     lr=lr, batch_size=batch_size, fix_seed=fix_seed)


if __name__ == '__main__':
    batched_comparison2()
    # traj_box_world_batched_main()
