from parser import parse_arguments

import matplotlib.pyplot as plt
import torch
import torchvision

from dataset import BIOSTEC2018
from torchvision import transforms
from transformations import named_transformations
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix(x, beta=1):
    lam = np.random.beta(beta, beta)
    split = x.shape[0] // 2
    x1, x2 = x[:split], x[split:]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_mix = x1.clone()
    x_mix[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x_mix


def no_op(x):
    return x


def mixup(x):
    """
    Mix the first half of the image batch with its second half
    e.g. for 4 images this will output 2 images: mix(img_0, img_2) and mix(img_1, img_3)
    """
    split = x.shape[0] // 2
    x1, x2 = x[:split], x[split:]
    lam = torch.Tensor([0.5 for _ in range(split)])
    for i in range(1, 4):
        lam = lam.unsqueeze(i)
    x_mix = lam * x1 + (1 - lam) * x2
    return x_mix


def visualize(dataset, num_images, transformations):
    for t in transformations:
        imgs = torch.stack(
            [t[1](dataset[idx][0]) for idx in range(num_images)],
            dim=0,
        )

        if t[0] == "Mixup":
            imgs = mixup(imgs)
        if t[0] == "CutMix":
            imgs = cut_mix(imgs)

        img_grid = torchvision.utils.make_grid(
            imgs, nrow=6, normalize=True, pad_value=0.9
        )
        img_grid = img_grid.permute(1, 2, 0)

        plt.figure(figsize=(10, 5))
        plt.title(t[0])
        plt.imshow(img_grid)
        plt.axis("off")

    plt.show()
    plt.close()


if __name__ == "__main__":
    args = parse_arguments(mode="visualize")
    dataset = BIOSTEC2018(
        path=args.dataset_dir,
        split="train",
        transform=transforms.Compose(
            [
                transforms.Resize(150),
                transforms.ConvertImageDtype(torch.float32),
            ]
        ),
    )
    transformations = [
        # *named_transformations,
        ("Mixup", no_op),
        ("CutMix", no_op),
        ("Originals", no_op),
    ]
    visualize(dataset, args.n, transformations)
