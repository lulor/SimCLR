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
    loss, top_acc = criterion(feats)

    if optimizer is not None:  # i.e. training mode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), top_acc


def train(model, criterion, optimizer, train_dl, device):

    model.train()
    criterion.enforce_batch_size()  # all batches have the same size

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0

    pbar = tqdm(train_dl, ncols=100)
    for imgs, _ in pbar:
        loss, top_acc = step(imgs, device, model, criterion, optimizer)
        running_loss += loss
        (top1_sum, top5_sum), batch_size = top_acc
        top1_acc = top1_sum / batch_size
        top5_acc = top5_sum / batch_size
        running_top1 += top1_acc
        running_top5 += top5_acc
        pbar.set_postfix_str(utils.postfix_str(top1_acc, top5_acc, loss))

    logging.info(
        "Training complete [avg_loss: %.4f, top1: %.2f, top5: %.2f]",
        running_loss / len(train_dl),
        running_top1 / len(train_dl) * 100,
        running_top5 / len(train_dl) * 100,
    )


def val(model, criterion, val_dl, device):

    model.eval()
    criterion.enforce_batch_size(False)

    n_samples = 0
    running_top1 = 0
    running_top5 = 0

    pbar = tqdm(val_dl, ncols=70)
    for imgs, _ in pbar:
        _, (top_sums, batch_size) = step(imgs, device, model, criterion)
        n_samples += batch_size
        top1_sum, top5_sum = top_sums
        top1_acc = top1_sum / batch_size
        top5_acc = top5_sum / batch_size
        running_top1 += top1_sum
        running_top5 += top5_sum
        pbar.set_postfix_str(utils.postfix_str(top1_acc, top5_acc))

    val_top5_acc = running_top5 / n_samples * 100

    logging.info(
        "Validation complete [top1: %.2f, top5: %.2f]",
        running_top1 / n_samples * 100,
        val_top5_acc,
    )

    return val_top5_acc


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

    train_ds, val_ds = utils.split_dataset(
        BIOSTEC2018(
            args.dataset_dir,
            split="train",
            transform=ContrastiveTransformations(train_transformations(ds_stats)),
        ),
        split=0.8,
    )

    train_dl = utils.build_data_loader(train_ds, args.batch_size, mode="train")
    val_dl = utils.build_data_loader(val_ds, args.batch_size, mode="eval")

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

    ### Model resume ###

    if args.reload is not None:
        checkpoint = utils.load_checkpoint(args.reload)
        epoch, best_val_top5_acc = utils.resume_from_state(checkpoint, model, optimizer)
        first_epoch = epoch + 1
        logging.info(
            "Checkpoint reloaded (epoch: %i, best_val_top5_acc: %.2f)\n",
            epoch,
            best_val_top5_acc,
        )
    else:
        best_val_top5_acc = 0.0
        first_epoch = 1

    ### Training loop ###

    not_improved = 0

    for epoch in range(first_epoch, args.epochs + 1):

        logging.info("Epoch [%i/%i]", epoch, args.epochs)
        train(model, criterion, optimizer, train_dl, args.device)
        val_top5_acc = val(model, criterion, val_dl, args.device)

        if val_top5_acc > best_val_top5_acc:
            best_val_top5_acc = val_top5_acc
            not_improved = 0
            logging.info("Improved\n")
        else:
            not_improved += 1
            logging.info("Not improving for %i epochs\n", not_improved)

        state = utils.build_state(model, optimizer, best_val_top5_acc, epoch)
        utils.save_checkpoint(args.out_dir, state, not_improved == 0, "last_model.pth")


if __name__ == "__main__":
    main()
