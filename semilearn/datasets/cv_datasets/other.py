# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import math
import torchvision
from torchvision import datasets
from torchvision.datasets import folder as dataset_parser, StanfordCars
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_ssl_data, split_images_labels


def get_other_dset(args, alg, dataset, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    # fixme: customize
    # data_dir = os.path.join(data_dir, 'semi_fgvc')

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    # fixme: customize
    # train_labeled_dataset = iNatDataset(alg, data_dir, train_split, dataset, transform=transform_weak, transform_strong=transform_strong)
    # train_unlabeled_dataset = iNatDataset(alg, data_dir, ulb_split, dataset, is_ulb=True, transform=transform_weak, transform_medium=transform_medium, transform_strong=transform_strong)
    # test_dataset = iNatDataset(alg, data_dir, 'test', dataset, transform=transform_val)

    if dataset == "sun397":
        dset = datasets.SUN397("/home/lhz/data/sun397", download=False)
        data, targets = dset._image_files, dset._labels

        data, targets = np.asarray(data), np.asarray(targets)

        train_data, train_targets = [], []
        test_data, test_targets = [], []
        for i in range(397):
            class_idx = np.where(targets == i)[0]
            class_sample_num = np.size(class_idx)
            test_idx = np.random.choice(class_idx, size=int(class_sample_num * 0.2), replace=False)
            train_idx = class_idx[~np.isin(class_idx, test_idx)]

            train_data.append(data[train_idx])
            train_targets.append(targets[train_idx])
            test_data.append(data[test_idx])
            test_targets.append(targets[test_idx])

        train_data, train_targets = np.concatenate(train_data, axis=0), np.concatenate(train_targets, axis=0)
        test_data, test_targets = np.concatenate(test_data, axis=0), np.concatenate(test_targets, axis=0)

    elif dataset == "cub":
        dataset_dir = "/home/lhz/data/CUB_200_2011/images"
        dset = datasets.ImageFolder(dataset_dir)
        data, targets = split_images_labels(dset.imgs)

        train_data, train_targets = [], []
        test_data, test_targets = [], []
        for i in range(200):
            class_idx = np.where(targets == i)[0]
            class_sample_num = np.size(class_idx)
            test_idx = np.random.choice(class_idx, size=int(class_sample_num * 0.2), replace=False)
            train_idx = class_idx[~np.isin(class_idx, test_idx)]

            train_data.append(data[train_idx])
            train_targets.append(targets[train_idx])
            test_data.append(data[test_idx])
            test_targets.append(targets[test_idx])

        train_data, train_targets = np.concatenate(train_data, axis=0), np.concatenate(train_targets, axis=0)
        test_data, test_targets = np.concatenate(test_data, axis=0), np.concatenate(test_targets, axis=0)

    elif dataset == "stanfordcars":
        dataset_dir = "/home/lhz/data"
        train_dset = StanfordCars(dataset_dir, split='train', download=False)
        test_dset = StanfordCars(dataset_dir, split='test', download=False)

        train_data = [t[0] for t in train_dset._samples]
        train_targets = [t[1] for t in train_dset._samples]

        test_data = [t[0] for t in test_dset._samples]
        test_targets = [t[1] for t in test_dset._samples]

    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_data, train_targets, num_classes,
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    lb_dset = OtherDataset(alg, dataset, transform=transform_weak, transform_medium=transform_medium,
                           transform_strong=transform_strong, samples=lb_data, targets=lb_targets,
                           num_classes=num_classes)
    ulb_dset = OtherDataset(alg, dataset, is_ulb=True, transform=transform_weak,
                            transform_medium=transform_medium, transform_strong=transform_strong,
                            samples=ulb_data, targets=ulb_targets, num_classes=num_classes)
    eval_dset = OtherDataset(alg, dataset, transform=transform_val, samples=test_data,
                             targets=test_targets, num_classes=num_classes)

    return lb_dset, ulb_dset, eval_dset


class OtherDataset(BasicDataset):
    def __init__(self, alg, dataset, transform=None, transform_medium=None, transform_strong=None,
                 loader=dataset_parser.default_loader, is_ulb=False, samples=None, targets=None, num_classes=None,
                 *args, **kwargs):
        super().__init__(alg, samples, targets=targets)
        self.alg = alg
        self.is_ulb = is_ulb
        self.loader = loader
        self.dataset = dataset
        self.samples, self.num_classes, self.targets = samples, num_classes, targets

        self.transform = transform
        self.medium_transform = transform_medium
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"

        self.strong_transform = transform_strong
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher',
                                        'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i])

    def __sample__(self, idx):
        # path, target = self.samples[idx]
        if self.dataset in ["sun397", "cub", "stanfordcars"]:
            img = self.loader(self.samples[idx])
        else:
            img = self.samples[idx]
        target = self.targets[idx]
        return img, target

    def __len__(self):
        return len(self.data)
