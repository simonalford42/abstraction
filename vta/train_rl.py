import argparse
import sys
import os
import logging
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
from torch.optim import Adam
from grid_world import grid
import bw_utils
import box_world

# from torch.utils.tensorboard import SummaryWriter
from hssm_rl import EnvModel
import utils
from utils import (
    preprocess,
    postprocess,
    full_dataloader,
    full_dataloader_toy,
    gridworld_loader,
    state_independent_loader,
    state_dependent_loader,
    log_train,
    log_test,
    plot_rec,
    plot_gen,
)
import modules
from modules import (
    GridEncoder,
    GridDecoder,
    GridActionEncoder,
    Encoder,
    StateIndependentEncoder,
    StateDependentEncoder
)
from datetime import datetime
import wandb

LOGGER = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="vta agr parser")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--name", type=str, default="st")

    # data size
    parser.add_argument("--dataset-path", type=str, default="./data/demos")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=1)

    # model size
    parser.add_argument("--state-size", type=int, default=8)
    parser.add_argument("--belief-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)

    # observation distribution
    parser.add_argument("--obs-std", type=float, default=1.0)
    parser.add_argument("--obs-bit", type=int, default=5)

    # optimization
    parser.add_argument("--learn-rate", type=float, default=0.0005)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=100000)

    # subsequence prior params
    parser.add_argument("--seg-num", type=int, default=100)
    parser.add_argument("--seg-len", type=int, default=100)

    # gumbel params
    parser.add_argument("--max-beta", type=float, default=1.0)
    parser.add_argument("--min-beta", type=float, default=0.1)
    parser.add_argument("--beta-anneal", type=float, default=100)

    # log dir
    parser.add_argument("--log-dir", type=str, default="./log/")

    # coding length params
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--rec_coeff", type=float, default=1.0)
    parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    parser.add_argument("--use_min_length_boundary_mask", action="store_true")

    # use second encoder for boxworld so that attention net isnt used for action decoding
    parser.add_argument("--second_enc", action="store_true")
    # use the attention net for boxworld encoder
    parser.add_argument("--attention_enc", action="store_true")

    return parser.parse_args()


def date_str():
    s = str(datetime.now())
    d, t = s.split(" ")
    t = "-".join(t.split(":")[:-1])
    return d + "-" + t


def set_exp_name(args):
    exp_name = args.name + "_" + date_str()
    return exp_name


def main():
    # parse arguments
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    init_size = args.init_size

    # set device as gpu
    device = torch.device("cuda", 0)
    # device = torch.device("cpu")

    # set writer
    exp_name = set_exp_name(args)

    # writer = SummaryWriter(args.log_dir + exp_name)
    wandb.init(
        project="vta",
        entity="simonalford42",
        name=exp_name,
        settings=wandb.Settings(start_method="fork"),
    )

    LOGGER.info("EXP NAME: " + exp_name)
    LOGGER.info(">"*80)
    LOGGER.info(args)
    LOGGER.info(">"*80)

    encoder2 = None  # Default is no second encoder

    # load dataset
    if 'demos' in args.dataset_path:
        train_loader, test_loader = gridworld_loader(
            args.batch_size, path=args.dataset_path
        )
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size, embedding_size=args.belief_size
        )
        encoder = GridEncoder(embed_dim=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        LOGGER.info("gridworld Dataset loaded")
    elif "state_independent" in args.dataset_path:
        train_loader, test_loader = state_independent_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size, embedding_size=args.belief_size
        )
        encoder = StateIndependentEncoder(feat_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
    elif "state_dependent" in args.dataset_path:
        train_loader, test_loader = state_dependent_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size, embedding_size=args.belief_size
        )
        encoder = StateDependentEncoder(feat_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
    elif "compile" in args.dataset_path:
        train_loader, test_loader = utils.compile_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size
        )
        encoder = modules.CompILEGridEncoder(feat_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
    elif "boxworld" in args.dataset_path:
        train_loader, test_loader = utils.boxworld_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size
        )

        if args.attention_enc:
            encoder = modules.BwGridEncoder(input_dim=12, input_shape=(14,14), feat_size=args.belief_size)
        else:
            encoder = modules.CompILEGridEncoder(input_dim=12, input_shape=(14,14), feat_size=args.belief_size)

        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        if args.second_enc:
            encoder2 = modules.CompILEGridEncoder(input_dim=12, input_shape=(14,14), feat_size=args.belief_size)
    else:
        raise ValueError(f"Unrecognize dataset_path {args.dataset_path}")

    seq_size = train_loader.dataset.seq_size
    if hasattr(train_loader.dataset, 'init_size'):
        args.init_size = train_loader.dataset.init_size

    wandb.config.params = args
    args.id = wandb.run.id

    # init models
    model = EnvModel(
        action_encoder=action_encoder,
        encoder=encoder,
        decoder=decoder,
        belief_size=args.belief_size,
        state_size=args.state_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        kl_coeff=args.kl_coeff,
        rec_coeff=args.rec_coeff,
        use_abs_pos_kl=args.use_abs_pos_kl == 1.0,
        coding_len_coeff=args.coding_len_coeff,
        use_min_length_boundary_mask=args.use_min_length_boundary_mask,
        encoder2=encoder2,
    ).to(device)
    LOGGER.info("Model initialized")

    # init optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

    # test data
    if test_loader.dataset.provides_true_boundaries:
        pre_test_full_state_list, pre_test_full_action_list, pre_test_full_true_boundaries = next(iter(test_loader))
        pre_test_full_true_boundaries = pre_test_full_true_boundaries.to(device)
    else:
        pre_test_full_state_list, pre_test_full_action_list = next(iter(test_loader))
    pre_test_full_state_list = pre_test_full_state_list.to(device)
    pre_test_full_action_list = pre_test_full_action_list.to(device)

    # for each iter
    b_idx = 0
    while b_idx <= args.max_iters:
        # for each batch
        for batch in train_loader:
            if train_loader.dataset.provides_true_boundaries:
                train_obs_list, train_action_list, true_boundaries = batch
            else:
                train_obs_list, train_action_list = batch


            b_idx += 1
            # mask temp annealing
            if args.beta_anneal:
                model.state_model.mask_beta = (
                    args.max_beta - args.min_beta
                ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            else:
                model.state_model.mask_beta = args.max_beta

            ##############
            # train time #
            ##############
            train_obs_list = train_obs_list.to(device)
            train_action_list = train_action_list.to(device)

            # run model with train mode
            model.train()
            optimizer.zero_grad()
            results = model(
                train_obs_list, train_action_list, seq_size, init_size, args.obs_std
            )

            if args.coding_len_coeff > 0:
                if results['obs_cost'].mean() < 0.05:
                    model.coding_len_coeff += 0.00002
                elif b_idx > 0:
                    model.coding_len_coeff -= 0.00002

                model.coding_len_coeff = min(0.005, model.coding_len_coeff)
                model.coding_len_coeff = max(0.000000, model.coding_len_coeff)
                results['coding_len_coeff'] = model.coding_len_coeff

            # get train loss and backward update
            train_total_loss = results["train_loss"]
            train_total_loss.backward()
            if args.grad_clip > 0.0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # log
            if b_idx % 10 == 0:
                results['grad_norm'] = grad_norm
                train_stats, log_str, log_data = log_train(results, None, b_idx)

                if not train_loader.dataset.provides_true_boundaries:
                    # Boundaries for grid world
                    true_boundaries = (
                            train_action_list[:, init_size:-init_size] == 4)
                    true_boundaries = torch.roll(true_boundaries, 1, -1)
                    true_boundaries[:, 0] = True

                correct_boundaries = torch.logical_and(
                    results["mask_data"].squeeze(-1) == true_boundaries,
                    true_boundaries).sum()
                num_pred_boundaries = results["mask_data"].sum()
                num_true_boundaries = true_boundaries.sum()
                train_stats["train/precision"] = (
                        correct_boundaries / num_pred_boundaries)
                train_stats["train/recall"] = (
                        correct_boundaries / num_true_boundaries)

                LOGGER.info(log_str, *log_data)
                wandb.log(train_stats, step=b_idx)

            np.set_printoptions(threshold=100000)
            torch.set_printoptions(threshold=100000)
            if b_idx % 200 == 0:
                exp_dir = os.path.join("experiments", args.name, str(b_idx))
                os.makedirs(exp_dir, exist_ok=True)
                for batch_idx in range(min(train_obs_list.shape[0], 10)):
                    states = train_obs_list[batch_idx][init_size:-init_size]
                    actions = train_action_list[batch_idx][init_size:-init_size]
                    reconstructed_actions = torch.argmax(
                            results["rec_data"], -1)[batch_idx]
                    options = results["option_list"][batch_idx]
                    boundaries = results["mask_data"][batch_idx]
                    frames = []
                    curr_option = options[0]
                    for seq_idx in range(states.shape[0]):
                        if boundaries[seq_idx].item() == 1:
                            curr_option = options[seq_idx]

                        if args.dataset == 'boxworld':
                            h, w = states.shape[1:3]
                            frame = grid.GridRender(width=w, height=h)
                            for y in range(h):
                                for x in range(w):
                                    obj = np.argmax(
                                            states[seq_idx][x][y].cpu().data.numpy())
                                    frame.draw_rectangle(np.array((x, y)), 0.4, box_world.COLOR_NAMES[obj])

                        frame = grid.GridRender(10, 10)
                        for x in range(10):
                            for y in range(10):
                                obj = np.argmax(
                                        states[seq_idx][x][y].cpu().data.numpy())
                                if obj == grid.ComPILEObject.num_types() or states[seq_idx][x][y][grid.ComPILEObject.num_types()]:
                                    frame.draw_rectangle(
                                            np.array((x, y)), 0.9, "cyan")
                                elif obj == grid.ComPILEObject.num_types() + 1:
                                    frame.draw_rectangle(
                                            np.array((x, y)), 0.7, "black")
                                elif states[seq_idx][x][y][obj] == 1:
                                    frame.draw_rectangle(
                                            np.array((x, y)), 0.4,
                                            grid.ComPILEObject.COLORS[obj])

                        frame.write_text(
                                f"Action: {repr(grid.Action(actions[seq_idx].item()))}")
                        frame.write_text(
                                f"Reconstructed: {repr(grid.Action(reconstructed_actions[seq_idx].item()))}")
                        if actions[seq_idx].item() == reconstructed_actions[seq_idx].item():
                            frame.write_text("CORRECT")
                        else:
                            frame.write_text("WRONG")

                        frame.write_text(f"Option: {curr_option}")
                        frame.write_text(f"Boundary: {boundaries[seq_idx].item()}")
                        frame.write_text(f"Obs NLL: {results['obs_cost'].mean()}")
                        frame.write_text(
                                f"Coding length: {results['encoding_length'].item()}")
                        frame.write_text(
                                f"Num reads: {results['mask_data'].sum(1).mean().item()}")
                        frames.append(frame.image())
                    save_path = os.path.join(exp_dir, f"{batch_idx}.gif")
                    frames[0].save(save_path, save_all=True,
                                   append_images=frames[1:], duration=750,
                                   loop=0, optimize=True, quality=20)

            if b_idx % 100 == 0:
                LOGGER.info("#" * 80)
                LOGGER.info(">>> option list")
                LOGGER.info("\n" + repr(results["option_list"][:10]))
                LOGGER.info(">>> boundary mask list")
                LOGGER.info("\n" + repr(results["mask_data"][:10].squeeze(-1)))
                # LOGGER.info("\n" + repr(torch.stack(results["all_boundaries"])))
                LOGGER.info(">>> train_action_list")
                LOGGER.info("\n" + repr(train_action_list[:10]))
                LOGGER.info(">>> argmax reconstruction")
                LOGGER.info("\n" + repr(torch.argmax(results["rec_data"], -1)[:10]))
                LOGGER.info(">>> diff")
                LOGGER.info("\n" + repr(train_action_list[:10, 1:-1] - torch.argmax(results["rec_data"][:10], -1)))
                LOGGER.info(">>> marginal")
                LOGGER.info("\n" + repr(results["marginal"]))
                # LOGGER.info(">>> s")
                # LOGGER.info("\n" + repr(results['pos_obs_state'][:, :, :5]))
                # LOGGER.info(">>> z")
                # LOGGER.info("\n" + repr(results['abs_state']))
                LOGGER.info(">>> softmax reconstruction")
                LOGGER.info("\n" + repr(nn.functional.softmax(results["rec_data"], -1).cpu().data.numpy().round(2)))
                LOGGER.info("#" * 80)

            if b_idx % 2000 == 0:
                exp_dir = os.path.join("experiments", str(args.id))
                bw_utils.save_model(model.state_model,
                                    os.path.join(exp_dir, f"model-{b_idx}.ckpt"))

            #############
            # test time #
            #############
            if b_idx % 100 == 0:
                with torch.no_grad():
                    ##################
                    # test data elbo #
                    ##################
                    model.eval()
                    results = model(
                        pre_test_full_state_list,
                        pre_test_full_action_list,
                        seq_size,
                        init_size,
                        args.obs_std,
                    )
                    post_test_rec_data_list = postprocess(
                        results["rec_data"], args.obs_bit
                    )

                    # log
                    test_stats, log_str, log_data = log_test(results, None, b_idx)

                    if not test_loader.dataset.provides_true_boundaries:
                        # Boundaries for grid world
                        true_boundaries = (
                                pre_test_full_action_list[:, init_size:-init_size] == 4)
                        true_boundaries = torch.roll(true_boundaries, 1, -1)
                        true_boundaries[:, 0] = True
                    else:
                        true_boundaries = pre_test_full_true_boundaries

                    correct_boundaries = torch.logical_and(
                        results["mask_data"].squeeze(-1) == true_boundaries,
                        true_boundaries).sum()
                    num_pred_boundaries = results["mask_data"].sum()
                    num_true_boundaries = true_boundaries.sum()
                    test_stats["valid/precision"] = (
                            correct_boundaries / num_pred_boundaries)
                    test_stats["valid/recall"] = (
                            correct_boundaries / num_true_boundaries)
                    LOGGER.info(log_str, *log_data)
                    wandb.log(test_stats, step=b_idx)


if __name__ == "__main__":
    main()
