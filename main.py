import up_right
import torch
import random
import utils
from utils import DEVICE
import abstract
from abstract2 import ControlNet, HMMNet
from modules import FC, RelationalDRLNet
import box_world
import argparse
import up_right
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
    net = HMMNet(abstract_policy_net)

    # utils.load_model(net, f'models/model_1-21__4.pt')
    abstract.train_abstractions(data, net, epochs=100, lr=1E-4)
    # utils.save_model(net, f'models/model_9-17.pt')
    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)

    eval_viterbi(net, eval_data,)
    # abstract.sample_trajectories(net, eval_data, full_abstract=False)


def box_world_sv2():
    env = box_world.BoxWorldEnv()
    data = box_world.BoxWorldDataset(env=env, n=50)

    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    def placeholder_fn(x=0):
        return None

    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII, num_attn_blocks=4, num_heads=4).to(DEVICE)
    utils.load_mlflow_model(relational_net, "fc3178b8b9b94314b4a259aa5ff8d22d")

    abstract_policy_net = abstract.ControlAPN(
        a=4,
        net=relational_net,
    )
    net = ControlNet(abstract_policy_net)

    abstract.train_abstractions(data, net, epochs=10, lr=1E-4)

    box_world.eval_model(relational_net, box_world.BoxWorldEnv(), n=100,)


def eval_viterbi(net: HMMNet, data: up_right.TrajData):
    for i, s_i, actions, points in zip(range(len(data.traj_states)), data.traj_states, data.traj_moves, data.points):
        (x, y, x_goal, y_goal) = points[0][0]
        moves = ''.join(data.trajs[i])
        print(f'{moves}')
        path = abstract2.viterbi(net, s_i, actions)
        print(''.join(map(str, path)))
        print('-'*10)


def box_world_main():
    parser = argparse.ArgumentParser(description='Abstraction')
    parser.add_argument("--cnn",
                        action="store_true",
                        dest="cnn")
    parser.add_argument("--test",
                        action="store_true",
                        dest="test")
    args = parser.parse_args()
    print(f"args: {args}")

    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    n = 5000
    epochs = 500
    num_test = 100
    test_every = 1

    # net = RelationalDRLNet(input_channels=box_world.NUM_ASCII).to(DEVICE)
    # utils.load_mlflow_model(net, "1537451d1ed84d089453e238d5d92011")
    # box_world.eval_model(net, box_world.BoxWorldEnv(),
    #                      renderer=lambda obs: box_world.render_obs(obs, color=True, pause=0.001))

    box_world_sv_train(n=n, epochs=epochs, drlnet=not args.cnn, num_test=num_test, test_every=test_every)


def box_world_main2():
    random.seed(1)
    torch.manual_seed(1)
    utils.print_torch_device()

    n = 10
    epochs = 5
    num_test = min(n, 100)
    test_every = 1

    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII, num_attn_blocks=4, num_heads=4).to(DEVICE)
    abstract_policy_net = abstract.ControlAPN(
        a=4,
        net=relational_net,
    )
    net = ControlNet(abstract_policy_net)

    abstract.box_world_sv_train2(net, n=n, epochs=epochs, num_test=num_test, test_every=test_every)



if __name__ == '__main__':
    # up_right_main()
    box_world_main2()
    # box_world_sv2()
