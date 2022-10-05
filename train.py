import logging
import os

import torch
from tqdm import tqdm

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
    running_top1_acc = 0.0
    running_top5_acc = 0.0

    pbar = tqdm(train_dl, ncols=100)
    for imgs, _ in pbar:
        loss, ((n_top1, n_top5), batch_size) = step(
            imgs, device, model, criterion, optimizer
        )
        running_loss += loss
        top1_acc = n_top1 / batch_size
        top5_acc = n_top5 / batch_size
        running_top1_acc += top1_acc
        running_top5_acc += top5_acc
        pbar.set_postfix_str(
            utils.postfix_str({"loss": loss, "top1": top1_acc, "top5": top5_acc})
        )

    logging.info(
        "Training complete [loss: %.3f, top1: %.3f, top5: %.3f]",
        running_loss / len(train_dl),
        running_top1_acc / len(train_dl),
        running_top5_acc / len(train_dl),
    )


@torch.no_grad()
def val(model, criterion, val_dl, device):

    model.eval()
    criterion.enforce_batch_size(False)

    tot_samples = 0
    tot_top1 = 0
    tot_top5 = 0

    pbar = tqdm(val_dl, ncols=70)
    for imgs, _ in pbar:
        _, ((n_top1, n_top5), batch_size) = step(imgs, device, model, criterion)
        tot_samples += batch_size
        tot_top1 += n_top1
        tot_top5 += n_top5
        top1_acc = n_top1 / batch_size
        top5_acc = n_top5 / batch_size
        pbar.set_postfix_str(utils.postfix_str({"top1": top1_acc, "top5": top5_acc}))

    epoch_top1_acc = tot_top1 / tot_samples
    epoch_top5_acc = tot_top5 / tot_samples

    logging.info(
        "Validation complete [top1: %.3f, top5: %.3f]", epoch_top1_acc, epoch_top5_acc
    )

    return epoch_top1_acc, epoch_top5_acc


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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ### Training resuming ###

    if args.reload is not None:
        checkpoint = utils.load_checkpoint(args.reload)
        epoch, best_val_top5_acc = utils.resume_from_state(checkpoint, model, optimizer)
        if best_val_top5_acc > 1.0:
            best_val_top5_acc /= 100  # For compatibility with old code
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
        _, val_top5_acc = val(model, criterion, val_dl, args.device)

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
