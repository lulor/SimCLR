import logging

from torch import optim
from tqdm.auto import tqdm

import utils
from arg_parser import parse_arguments
from biostec import BIOSTEC2018, ds_stats
from nt_xent import NTXent
from simclr import SimCLR
from transformations import ContrastiveTransformations, train_transformations


def step(imgs, device, model, criterion, optimizer=None):
    """
    Perform a step on the received data batch
    if optimizer is None, no backward pass will be performed
    """

    imgs = utils.move_iterable_to(imgs, device)

    _, feats = model(imgs)
    loss = criterion(feats)

    if optimizer is not None:  # i.e. training mode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train(model, criterion, optimizer, train_loader, val_loader, args):

    for epoch in range(args.first_epoch, args.epochs + 1):

        ### Training ###

        logging.info("Training epoch %i", epoch)
        model.train()
        criterion.enforce_batch_size()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, ncols=80)
        for imgs, _ in train_pbar:
            loss = step(imgs, args.device, model, criterion, optimizer)
            running_loss += loss
            train_pbar.set_postfix({"batch_loss": loss})

        logging.info(
            "Epoch %i completed [avg_loss: %f]\n",
            epoch,
            running_loss / len(train_loader),
        )

        ### Validation ###

        model.eval()
        criterion.enforce_batch_size(False)

        val_pbar = tqdm(val_loader, ncols=80)
        for imgs, _ in val_pbar:
            step(imgs, args.device, model, criterion)


def main():
    args = parse_arguments()
    if args.device is None:
        args.device = utils.get_device()

    args.out_dir = utils.get_run_out_dir(args.out_dir)
    utils.setup_logging(args.out_dir)
    logging.debug(args)
    logging.info("Using device %s\n", args.device)

    utils.make_deterministic(args.seed)

    ### Dataset and DataLoaders ###

    dataset = BIOSTEC2018(
        args.dataset_dir,
        split="train",
        transform=ContrastiveTransformations(train_transformations(ds_stats)),
    )
    train_ds, val_ds = utils.split_dataset(dataset, split=0.8)

    train_loader = utils.build_data_loader(train_ds, args.batch_size, mode="train")
    val_loader = utils.build_data_loader(val_ds, args.batch_size, mode="eval")

    ### Model, Loss and Optimizer ###

    model = SimCLR(
        encoder="resnet18",
        projection_dim=args.projection_dim,
    ).to(args.device)

    criterion = NTXent(args.temperature, args.batch_size, args.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ### Train ###

    args.first_epoch = 1
    train(model, criterion, optimizer, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
