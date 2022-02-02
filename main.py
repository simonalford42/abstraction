import up_right
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import utils
from utils import DEVICE
import abstract
from abstract2 import UnbatchedTrajNet, Controller, TrajNet, HMMTrajNet
from modules import FC, RelationalDRLNet, abstract_out_dim
import box_world
import argparse
import abstract2


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
    abstract.train_abstractions(data, net, epochs=100, lr=1E-4)
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

    abstract.box_world_sv_train(n=n, epochs=epochs, num_test=num_test, test_every=test_every, rounds=-1, lr=lr)


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
        batched=False
    )
    unbatched_traj_net = UnbatchedTrajNet(control_net)

    batched_control_net = Controller(
        a=4,
        b=1,
        net=relational_net,
        batched=True,
    )
    traj_net = TrajNet(batched_control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env, n=1, traj=True, batch_size=1)
    data = box_world.BoxWorldDataset(env, n=1, traj=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=box_world.traj_collate)

    total = 0
    for d in dataloader:
        negative_logp = traj_net(*d)
        total += negative_logp

    print(f'total: {total}')

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
        loss = - torch.sum(logps[range(len(actions)), actions])
        total4 += loss
    print(f'total4: {total4}')

    abstract.train_supervised(dataloader, relational_net, epochs=1)


def traj_box_world_batched_main():
    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    # standard: n = 5000, epochs = 100, num_test = 200, lr = 8E-4, rounds = 10
    hmm = False
    n = 20
    epochs = 600
    num_test = 20
    lr = 8E-4
    rounds = 1
    fix_seed = True



    if hmm:
        b = 20
        print('hmm training!')
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
        net = HMMTrajNet(control_net)
        batch_size=1
    else:
        print('traj-level training without hmm')
        relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                          num_attn_blocks=2,
                                          num_heads=4,
                                          out_dim=abstract_out_dim(a=4, b=1)).to(DEVICE)
        control_net = Controller(
            a=4,
            b=1,
            net=relational_net,
            batched=True,
        )
        net = TrajNet(control_net)
        batch_size=10

    net = net.to(DEVICE)
    abstract.traj_box_world_sv_train(net, n=n, epochs=epochs,
            num_test=num_test, test_every=1, rounds=rounds, lr=lr,
            batch_size=batch_size, fix_seed=fix_seed)


if __name__ == '__main__':
    # up_right_main()
    # box_world_main()
    # batched_comparison()
    print('batch norm')
    traj_box_world_batched_main()
