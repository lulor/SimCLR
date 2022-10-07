from copy import deepcopy

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

import utils
from arg_parser import parse_arguments
from biostec import BIOSTEC2018, ds_stats
from simclr import SimCLR
from transformations import test_transformations


def step(imgs, labels, device, model, criterion, optimizer=None):
    """
    Perform a step on the received data batch
    if optimizer is None, no backward pass will be performed
    """

    imgs = imgs.to(device)
    labels = labels.to(device)

    scores = model(imgs)
    loss = criterion(scores, labels)

    if optimizer is not None:  # i.e. training mode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_correct = (scores.argmax(dim=1) == labels).sum().item()

    return loss.item(), (num_correct, imgs.size(0))


def train(model, criterion, optimizer, train_dl, device):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(train_dl, ncols=80)
    for imgs, labels in pbar:
        loss, (n_correct, batch_size) = step(
            imgs, labels, device, model, criterion, optimizer
        )
        running_loss += loss
        acc = n_correct / batch_size
        running_acc += acc
        pbar.set_postfix_str(utils.postfix_str({"loss": loss, "acc": acc}))

    epoch_loss = running_loss / len(train_dl)
    epoch_acc = running_acc / len(train_dl)

    print(
        f"Training complete [loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}]",
    )


@torch.no_grad()
def test(model, criterion, eval_dl, device):
    model.eval()

    tot_samples = 0
    tot_correct = 0

    pbar = tqdm(eval_dl, ncols=50)
    for imgs, labels in pbar:
        _, (n_correct, batch_size) = step(imgs, labels, device, model, criterion)
        tot_samples += batch_size
        tot_correct += n_correct
        acc = n_correct / batch_size
        pbar.set_postfix_str(utils.postfix_str({"acc": acc}))

    epoch_acc = tot_correct / tot_samples

    print(f"Eval complete [acc: {epoch_acc:.3f}]")

    return epoch_acc


@torch.no_grad()
def extract_features(encoder, ds_dict, batch_size, device):
    encoder.eval()

    feats_dict = {}

    for ds_name, dataset in ds_dict.items():
        loader = utils.build_data_loader(dataset, batch_size, "eval")

        feats_list = []
        labels_list = []

        print(f"Exctacting features for {ds_name} dataset")

        pbar = tqdm(loader, ncols=50)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            (feats,), _ = encoder((imgs,))
            feats_list.append(feats.cpu())
            labels_list.append(labels)

        feats_dict[ds_name] = TensorDataset(
            torch.cat(feats_list, dim=0),
            torch.cat(labels_list, dim=0),
        )

    return feats_dict


def main():

    args = parse_arguments("logreg")
    if args.device is None:
        args.device = utils.get_device()

    print(args)
    print(f"Using device {args.device}\n")

    utils.make_deterministic(args.seed)

    ### Datasets ###

    ds_dict = {}

    ds_dict["train"], ds_dict["val"] = utils.split_dataset(
        BIOSTEC2018(
            path=args.dataset_dir,
            split="train",
            transform=test_transformations(ds_stats),
        ),
        split=0.8,
    )
    ds_dict["test"] = BIOSTEC2018(
        path=args.dataset_dir, split="test", transform=test_transformations(ds_stats)
    )

    num_classes = 3

    ### Feature extraction ###

    encoder = SimCLR(encoder=args.encoder, projection_features=None).to(args.device)
    state = utils.load_checkpoint(args.reload, args.device)
    utils.reload_simclr_encoder(state, encoder)
    feats_dict = extract_features(
        encoder, ds_dict, args.encoder_batch_size, args.device
    )

    ### DataLoaders ###

    train_dl = utils.build_data_loader(feats_dict["train"], args.batch_size, "train")
    val_dl = utils.build_data_loader(feats_dict["val"], args.batch_size, "eval")
    test_dl = utils.build_data_loader(feats_dict["test"], args.batch_size, "eval")

    ### Model, Loss and Optimizer ###

    model = torch.nn.Linear(
        in_features=encoder.encoder_out_features,
        out_features=num_classes,
        device=args.device,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ### Training loop ###

    best_model = None
    best_val_acc = 0.0

    print()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        train(model, criterion, optimizer, train_dl, args.device)
        val_acc = test(model, criterion, val_dl, args.device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)
        print()

    print("Training complete\n")

    ### Eval on test set ###

    test(best_model, criterion, test_dl, args.device)


if __name__ == "__main__":
    main()
