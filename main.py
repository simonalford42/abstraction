import up_right
import torch
import random
import utils
from utils import assertEqual
import abstract
from abstract2 import HMMNet
from modules import FC, AllConv, RelationalDRLNet
import box_world


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

    # utils.load_model(net, f'models/model_9-10_{model}.pt')
    abstract.train_abstractions(data, net, epochs=100)
    # utils.save_model(net, f'models/model_9-17.pt')
    eval_data = up_right.TrajData(up_right.generate_data(scale, seq_len, n=10),
                                  max_coord=data.max_coord)
    abstract.sample_trajectories(net, eval_data, full_abstract=False)


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

    n = 1
    epochs = 500
    num_test = min(n, 100)
    test_every = 1

    net = RelationalDRLNet(input_channels=box_world.NUM_ASCII, num_attn_blocks=4, num_heads=4).to(DEVICE)
    utils.load_mlflow_model(net, "fc3178b8b9b94314b4a259aa5ff8d22d")
    box_world.eval_model(net, box_world.BoxWorldEnv(), n=500,
                         renderer=lambda obs: box_world.render_obs(obs, color=True, pause=0.001))

    # box_world_sv_train(n=n, epochs=epochs, drlnet=not args.cnn, num_test=num_test, test_every=test_every)


if __name__ == '__main__':
    up_right_main()
