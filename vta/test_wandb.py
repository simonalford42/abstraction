import wandb


def main():
    wandb.init(
        project="abstraction",
        entity="simonalford42",
    )

    # for each iter
    b_idx = 0
    while b_idx <= 100:
        # wandb.log({'loss': b_idx}, step=b_idx)
        b_idx += 1
    print('done')


if __name__ == "__main__":
    main()
