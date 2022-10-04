import logging
import os

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
    loss, sums = criterion(feats)

    if optimizer is not None:  # i.e. training mode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), sums


def train(model, criterion, optimizer, train_loader, val_loader, args):

    not_improved = 0
    best_val_top5_acc = 0.0

    for epoch in range(args.first_epoch, args.epochs + 1):

        logging.info("Epoch [%i/%i]", epoch, args.epochs)

        ### Training ###

        model.train()
        criterion.enforce_batch_size()  # all batches have the same size

        running_loss = 0.0
        running_top1 = 0.0
        running_top5 = 0.0

        pbar = tqdm(train_loader, ncols=100)
        for imgs, _ in pbar:
            loss, sums = step(imgs, args.device, model, criterion, optimizer)
            running_loss += loss
            top1_sum, top5_sum = sums
            top1_acc = top1_sum / (train_loader.batch_size * 2)
            top5_acc = top5_sum / (train_loader.batch_size * 2)
            running_top1 += top1_acc
            running_top5 += top5_acc
            pbar.set_postfix_str(utils.postfix_str(top1_acc, top5_acc, loss))

        logging.info(
            "Training complete [avg_loss: %.4f, top1: %.2f, top5: %.2f]",
            running_loss / len(train_loader),
            running_top1 / len(train_loader) * 100,
            running_top5 / len(train_loader) * 100,
        )

        ### Validation ###

        model.eval()
        criterion.enforce_batch_size(False)

        running_top1 = 0.0
        running_top5 = 0.0
        n_samples = 0

        pbar = tqdm(val_loader, ncols=70)
        for imgs, _ in pbar:
            batch_size = 2 * imgs[0].size(0)  # because of the augmentation
            n_samples += batch_size
            _, (top1_sum, top5_sum) = step(imgs, args.device, model, criterion)
            top1_acc = top1_sum / batch_size
            top5_acc = top5_sum / batch_size
            running_top1 += top1_sum
            running_top5 += top5_sum
            pbar.set_postfix_str(utils.postfix_str(top1_acc, top5_acc))

        val_top5_acc = running_top5 / n_samples * 100

        logging.info(
            "Validation complete [top1: %.2f, top5: %.2f]\n",
            running_top1 / n_samples * 100,
            val_top5_acc,
        )

        if val_top5_acc < best_val_top5_acc:
            not_improved += 1
            logging.info("Not improving since %i epochs", not_improved)


def main():
    args = parse_arguments()
    if args.device is None:
        args.device = utils.get_device()

    args.out_dir = utils.get_run_out_dir(args.out_dir)
    utils.setup_logging(os.path.join(args.out_dir, "logs"))
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
        encoder=args.encoder,
        projection_features=args.projection_dim,
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
