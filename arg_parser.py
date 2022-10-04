import argparse


def parse_arguments(mode="simclr"):
    parser = argparse.ArgumentParser(
        description="SimCLR", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_dir", type=str, help="Dataset path")

    if mode == "visualize":
        parser.add_argument(
            "--n",
            type=int,
            default=6,
            help="number of images to visualize",
        )

    elif mode == "baseline":
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--early_stopping", action="store_true")
        parser.set_defaults(early_stopping=False)
        parser.add_argument("--freeze_layers", action="store_true")
        parser.set_defaults(freeze_layers=False)

    elif mode == "simclr":
        parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
        parser.add_argument("--reload", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--projection_dim", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=0.07)
        parser.add_argument("--epochs", type=int, default=300)
        parser.add_argument("--out_dir", type=str, default=".")

        parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda", "mps", None],
            default=None,
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=5e-04,
            help="Learning rate",
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-04,
            help="Weight Decay",
        )

        parser.add_argument("--mixup", action="store_true")
        parser.set_defaults(mixup=False)

        parser.add_argument("--cutmix", action="store_true")
        parser.set_defaults(cutmix=False)

        # LogReg

        parser.add_argument(
            "--logistic_lr",
            type=float,
            default=1e-05,
            help="LogReg Learning rate",
        )

        parser.add_argument(
            "--logistic_weight_decay",
            type=float,
            default=1e-03,
            help="LogReg Weight Decay",
        )

        parser.add_argument("--logistic_epochs", type=int, default=100)
        parser.add_argument("--logistic_batch_size", type=int, default=64)

    else:
        raise RuntimeError(f"Illegal mode received: {mode}")

    args = parser.parse_args()

    return args
