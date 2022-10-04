import os
from glob import glob

from torch.utils.data import Dataset
from torchvision.io import read_image

ds_stats = (
    [0.6586, 0.4506, 0.5956],  # mean
    [0.1884, 0.2241, 0.1810],  # std
)


class BIOSTEC2018(Dataset):
    def __init__(self, path, split="train", transform=None):
        imgs_path = os.path.join(path, split)
        self.transform = transform
        self.data = []
        self.targets = []
        self.class_names = []
        class_names_set = set()
        file_list = glob(imgs_path + "/*")
        for class_idx, class_path in enumerate(file_list):
            class_name = class_path.split("/")[-1]
            self.class_names.append(class_name)
            class_names_set.add(class_name)
            for img_path in glob(class_path + "/*.png"):
                self.data.append(img_path)
                self.targets.append(class_idx)
        assert len(class_names_set) == len(self.class_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.targets[idx]
        image = read_image(img_path)

        view = image if self.transform is None else self.transform(image)

        return view, label
