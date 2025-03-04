import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=[128, 128], cropsize=128):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(self.resize),
                                        T.CenterCrop(cropsize),
                                        T.ToTensor(),
                                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        mask_i = 0

        if y == 0:
            mask_i = torch.zeros([1, self.resize[0], self.resize[0]])
        else:
            for id_mask in range(len(mask)):
                if id_mask == 0:
                    mask_i = self.transform_mask(Image.open(mask[id_mask]))
                else:
                    mask_i = mask_i + self.transform_mask(Image.open(mask[id_mask]))
        mask = mask_i
        mask[mask != 0] = 1
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)

                num_mask = sorted(os.listdir(gt_type_dir))
                for num_i in num_mask:
                    gt_fpath_list = []
                    num_mask_l = sorted(os.listdir(gt_type_dir + '/' + num_i))
                    for num_i_l in num_mask_l:
                        gt_fpath_list += [os.path.join(gt_type_dir, num_i, num_i_l)]
                    mask.append(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
