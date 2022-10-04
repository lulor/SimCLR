import random
import numpy as np
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F


class SobelFilter(object):
    def __init__(self, p=0.5):
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.p = p

    def get_gray(self, x):
        """
        Convert image to its gray one.
        """
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray
        # return x_gray.unsqueeze(1)

    def __call__(self, x):
        p = np.random.uniform(0, 1)
        if p > self.p:
            return x

        # print(f"Pre: {x.size()}")
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[0] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        # print(f"Before stack: {x.size()}")

        x = torch.cat((x, x, x), dim=0)

        # print(f"After stack: {x.size()}")

        return x


class Cutout(object):
    def __init__(self, mask_size, cutout_inside, p=0.5, mask_color=(0, 0, 0)):
        self.mask_size = mask_size
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0
        self.cutout_inside = cutout_inside
        self.p = p
        self.mask_color = mask_color

    def __call__(self, image):
        image = np.asarray(image).copy()

        if random.uniform(0, 1) > self.p:
            return image

        h, w = image.shape[:2]

        if self.cutout_inside:
            cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half
            cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        # yrnd = np.random.randint(self.mask_size_half)
        # xrnd = np.random.randint(self.mask_size_half)
        yrnd = 0
        xrnd = 0

        image[ymin + yrnd : ymax, xmin + xrnd : xmax] = self.mask_color

        return image


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, p=0.5, c=0.1):
        self.std = std
        self.mean = mean
        self.p = p
        self.c = c

    def __call__(self, tensor):
        out = tensor
        if random.uniform(0, 1) < self.p:  # apply
            out += self.c * (torch.randn(tensor.size()) * self.std + self.mean)
        return out


def grid_mask(use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0):
    st_prob = prob

    def _set_prob(epoch, max_epoch):
        prob = st_prob * epoch / max_epoch

    def _grid_mask(sample):
        nonlocal offset
        img = sample
        # img = np.asarray(img).copy()
        if np.random.rand() > prob:
            return sample
        # h = img.shape[0]
        # w = img.shape[1]
        h = img.shape[1]
        w = img.shape[2]
        d1 = 2
        d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(d1, d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if ratio == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        if mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        # added to match our dataset dimensions
        mask = np.moveaxis(mask, -1, 0)
        if offset:
            offset = float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        return img

    return _grid_mask


def train_transformations(ds_stats):
    ds_mean, ds_std = ds_stats
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=150),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            Cutout(40, cutout_inside=True, p=0.8),
            transforms.ToTensor(),
            # GaussianNoise(),
            # transforms.GaussianBlur(kernel_size),
            # AutoAugment(),
            transforms.Normalize(ds_mean, ds_std),
        ]
    )


def test_transformations(ds_stats):
    ds_mean, ds_std = ds_stats
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std),
        ]
    )


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return tuple(self.base_transforms(x) for _ in range(self.n_views))


def _wrap(t):
    return transforms.Compose([transforms.ToPILImage(), t, transforms.ToTensor()])


named_transformations = [
    (
        "Horizontal flip",
        transforms.RandomHorizontalFlip(p=1),
    ),
    (
        "Color Jitter",
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    ),
    (
        "Gray scale",
        transforms.RandomGrayscale(p=1),
    ),
    (
        "Cutout",
        _wrap(Cutout(40, cutout_inside=True, p=1)),
    ),
    (
        "Gaussian Noise",
        GaussianNoise(p=1),
    ),
]
