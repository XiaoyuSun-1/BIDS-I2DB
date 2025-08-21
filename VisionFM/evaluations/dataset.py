# the dataset classes used in downstream tasks
import os
import glob
import random
import numpy as np
import torch
import pickle
import json
from torch.utils.data import Dataset
import utils
from utils import pil_loader, npy_loader
from torchvision import datasets

from PIL import Image
from transforms_retifluidnet import RetifluidTransforms

class SegImgs(Dataset):
    """
    The dataset for the downstream segmentation task.
    root: str, the dataset root dir
    dst_root: str, the save dir for the extracted features, used in feature extraction stage.
    split: str, the training split mode, training or test
    transform: the involved transformations, the data augmentation
    loader: func, the loader to load the image, default is the pil_loader
    few_shot: int, select the {few_shot} images from the training set.
    label_suiffix: str, the suffix of files in `labels` directory, default is png

    The folder structure should be:
    ├── /XXX/VesselSegmentation/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── images/
    │   │   │   │   ├── 1.jpg
    │   │   │   │   └── ...
    │   │   │   └── labels/
    │   │   │       ├── 1.png
    │   │   │       └── ...
    │   │   └── test/
    │   │       └── ...
    │   ├── dataset_B/
    │   │   └── ...
    │   └── ...
    └── ...

    The range of pixel values in images in labels directory should be [0, C-1], where C is the number of classes.

    the file name of label should be same as that of the corresponding image file.
    For example, label path: /xxx/VesselSegmentation/dataset_A/training/masks/1.png, image path: /xxx/VesselSegmentation/dataset_A/training/images/1.png
    Otherwise, you need to manually assign their corresponding relationships.
    """
    def __init__(self, root:str, split, dst_root:str=None, transform=None, loader=pil_loader, few_shot=-1, label_suiffix='png', split_file: str = None, fold: int = 0):
        self.root:str = root
        self.dst_root = dst_root # the dst root dir for extracted features
        self.split = split
        assert self.split in ["training", "test"], f"unsupported split mode: {self.split}"
        self.transform = transform
        self.loader = loader
        self.label_suffix = label_suiffix
        
        self.split_file = split_file  # added
        self.fold = fold             # added
        
        self.few_shot = few_shot

        self._entries = []
        self._mask_entries = [] # added: parallel mask list
        self._entries_cache = []

        # if we passed a split_file, skip glob and load straight from JSON
        if self.split_file:
            with open(self.split_file, 'r') as f:
                folds = json.load(f)
            side = 'train' if self.split == 'training' else 'val'
            info = folds[self.fold][side]
            base_dir = os.path.dirname(self.split_file)

            # build absolute paths by joining that base_dir with the JSON‐listed subpaths
            self._entries      = [os.path.join(base_dir, p) for p in info['images']]
            self._mask_entries = [os.path.join(base_dir, p) for p in info['masks']]

            print(f"[SegImgs] loaded {len(self._entries)} image/mask pairs from fold={self.fold}, split={side}")
        else:
            # glob‐based fallback
            assert os.path.exists(self.root), f"cannot find {self.root}"
            sub_datasets = utils.get_sub_dirs(self.root)
            for sd in sub_datasets:
                img_dir = os.path.join(sd, self.split, 'images')
                assert os.path.exists(img_dir)
                self._entries += glob.glob(os.path.join(img_dir, '*'))
            random.shuffle(self._entries)

        # few-shot subsampling for training only
        if self.few_shot != -1 and self.split == "training":
            self._entries = random.sample(self._entries, 1)
            print(f"select {self.few_shot} images for the few-shot setting")
        
        # determine how many variants RetifluidTransforms.train returns per image
        if self.split == 'training' and self.transform is RetifluidTransforms.train:
            sample_img  = Image.open(self._entries[0])
            sample_mask = (
                Image.open(self._mask_entries[0]).convert('L')
                if self._mask_entries
                else sample_img
            )
            out0 = self.transform(sample_img, sample_mask)
            self._n_augs = len(out0) if isinstance(out0, list) else 1
        else:
            self._n_augs = 1

        print(f"[SegImgs] using {self._n_augs} augmentations per image")    

    def get_entries(self, sub_datasts:list):
        # read all images
        for sub_dataset in sub_datasts:
            print(f"reading images at {sub_dataset}")
            img_dir = os.path.join(sub_dataset, self.split, 'images')
            assert os.path.exists(img_dir), f"cannot find the {img_dir}"
            img_files = glob.glob(os.path.join(img_dir, "*"))
            print(f"there are {len(img_files)} image files in {img_dir}")
            self._entries += img_files
        print(f"the total number of images is: {len(self._entries)}")

    def __len__(self):
        # training: each image produces _n_augs samples
        if self.split == 'training':
            return len(self._entries) * self._n_augs
        else:
            return len(self._entries)

    def __getitem__(self, idx):      
        # map global idx to image index, augmentation index
        if self.split=="training":
            img_idx = idx // self._n_augs
            aug_idx = idx % self._n_augs
        else:
            img_idx, aug_idx = idx, None

        img_path = self._entries[img_idx]        
        
        # load image
        try:
            img = self.loader(img_path)
        except BaseException as e:
            print(f"cannot load {img_path} due to the error: {e}")
            print(f"will randomly load another one.")
            index = random.randint(0, self.__len__())
            img_path = self._entries[index]
            img = self.loader(img_path)

        if self.dst_root is None:
            dst_path = None
        else:
            dst_path = img_path.replace(self.root, self.dst_root)

        # use mask list from JSON
        if self._mask_entries:
            label_path = self._mask_entries[img_idx]
        else:
            label_path = (
                img_path
                .replace('/retouch_data/', '/retouch_data_mask/')
                .replace('_oct_', '_mask_')
            )      

        if self.label_suffix != 'png':
            label_path = label_path.replace('.png', f'.{self.label_suffix}')
        assert os.path.exists(label_path), f"cannot find the {label_path}."

        if self.label_suffix == 'png':
            label = self.loader(label_path).convert('L')
        elif self.label_suffix == 'npy':
            label:np.ndarray = npy_loader(label_path) 
            label = torch.from_numpy(label).permute(2, 0, 1) 
 
        # apply transform
        if self.transform is not None:
            out = self.transform(img, label)
            if isinstance(out, list):
                img, label = out[aug_idx]
            else:
                img, label = out

        if self.dst_root is not None:
            extras = {'img_path': img_path, 'dst_path':dst_path}
        else:
            extras = {'img_path': img_path}
        return img, label, extras
