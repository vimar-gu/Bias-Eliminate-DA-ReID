# encoding: utf-8

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, size=None):
        self.dataset = dataset
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, setid, trkid, pidx = self.dataset[index]
        img = read_image(img_path)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, setid, trkid


class CamStyleImageDataset(Dataset):
    """CamStyle Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, size=None):
        self.dataset = dataset.train
        self.camstyle_dict = dataset.camstyle_dict
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, setid, trkid, pidx = self.dataset[index]
        
        img_list = []
        img = read_image(img_path)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        img_list.append(np.asarray(img))

        for fake_img_path in self.camstyle_dict[str(pid) + '_' + str(camid) + '_' + str(pidx)]:
            fake_img = read_image(fake_img_path)
            fake_img = fake_img.resize((self.size[1], self.size[0]), Image.BILINEAR)
            img_list.append(np.asarray(fake_img))
        
        img = np.mean(img_list, 0).astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, setid, trkid


class CameraImageDataset(Dataset):
    """Camera Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, size=None):
        self.dataset = dataset
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, setid, trkid, pidx = self.dataset[index]
        img = read_image(img_path)
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, camid, camid, img_path, setid, trkid
