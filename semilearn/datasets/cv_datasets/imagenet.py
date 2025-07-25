# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    data_dir = os.path.join(data_dir, name.lower())
    ratio = 0
    if ratio == 0:
        dataset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg)
        percentage = num_labels / len(dataset)

        lb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, percentage=percentage)

        ulb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, alg=alg, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

        eval_dset = ImagenetDataset(root=os.path.join(data_dir, "val"), transform=transform_val, alg=alg, ulb=False)
        return lb_dset, ulb_dset, eval_dset
    
    else:
        train_dset = ImageFolder(os.path.join(data_dir, "train"))
        train_targets = np.array(train_dset.imgs)[:, 1]
        train_targets = list(map(int, train_targets.tolist()))
        train_data_len = len(train_targets)
        lbl_percent = num_labels / train_data_len
        classes = list(range(num_classes))
        known_classes = list(range(round(num_classes * ratio)))
        novel_classes = list(set(classes) - set(known_classes))
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(train_targets, lbl_percent, num_classes, known_classes, novel_classes)
        
        lb_dset = GenericSSL(alg, os.path.join(data_dir, 'train'), train_labeled_idxs, num_classes,
                                           transform=transform_weak, medium_transform=transform_medium, strong_transform=transform_strong, is_ulb=False)
        if include_lb_to_ulb:
            ulb_dset = GenericSSL(alg, os.path.join(data_dir, 'train'), train_unlabeled_idxs + train_labeled_idxs, num_classes, 
                                             transform=transform_weak, medium_transform=transform_medium, strong_transform=transform_strong, is_ulb=True)
        else:
            ulb_dset = GenericSSL(alg, os.path.join(data_dir, 'train'), train_unlabeled_idxs, num_classes, 
                                             transform=transform_weak, medium_transform=transform_medium, strong_transform=transform_strong, is_ulb=True)
        eval_dset = GenericTEST(alg, os.path.join(data_dir, 'val'), num_classes, transform=transform_val)
        eval_novel_dset = GenericTEST(alg, os.path.join(data_dir, 'val'), num_classes, transform=transform_val, labeled_set=novel_classes)
        eval_known_dset = GenericTEST(alg, os.path.join(data_dir, 'val'), num_classes, transform=transform_val, labeled_set=known_classes)
        return lb_dset, ulb_dset, eval_dset, eval_novel_dset, eval_known_dset
    
    
class GenericSSL(BasicDataset, ImageFolder):
    def __init__(self, alg, root, indexs, num_classes, transform=None, medium_transform=None, strong_transform=None, 
                 is_ulb=False, temperature=None, temp_uncr=None, target_transform=None):
        ImageFolder.__init__(self, root, transform=transform, target_transform=target_transform)
        
        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)
        self.alg = alg
        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.transform = transform
        self.medium_transform = medium_transform
        self.strong_transform = strong_transform

        if temperature is not None:
            self.temp = temperature * np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        # BasicDataset.__init__(self, alg, self.data, self.targets, num_classes,
                                        #    is_ulb=is_ulb, transform=transform, medium_transform=medium_transform, strong_transform=strong_transform)

    def __len__(self):
        return len(self.data)

    def __sample__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)
        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class GenericTEST(BasicDataset, ImageFolder):
    def __init__(self, alg, root, num_classes, transform=None, is_ulb=False, target_transform=None, labeled_set=None):
        ImageFolder.__init__(self, root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)
        self.alg = alg
        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.transform = transform

        indexs = []
        if labeled_set is not None:
            for i in range(num_classes):
                if i in labeled_set:
                    idx = np.where(self.targets == i)[0]
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            
        pass
        # BasicDataset.__init__(self, alg, self.data, self.targets, num_classes, is_ulb=is_ulb, transform=transform)

    def __len__(self):
        return len(self.data)

    def __sample__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, medium_transform=None, strong_transform=None, percentage=-1, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)
        
        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.percentage != -1:
                    fnames = fnames[:int(len(fnames) * self.percentage)]
                if self.percentage != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if fname in self.lb_index[target_class]:
                            continue
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        gc.collect()
        self.lb_idx = lb_idx
        return instances

