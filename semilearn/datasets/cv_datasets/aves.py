# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import math
from torchvision import datasets
from torchvision.datasets import folder as dataset_parser
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_ssl_data, split_images_labels


def get_semi_aves(args, alg, dataset, num_labels, num_classes, train_split='l_train_val', ulb_split='u_train_in',
                  data_dir='./data', include_lb_to_ulb=True):
    assert train_split in ['l_train', 'l_train_val']

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

    # NOTE this dataset is inherently imbalanced with unknown distribution
    # fixme: customize
    # train_labeled_dataset = iNatDataset(alg, data_dir, train_split, dataset, transform=transform_weak, transform_strong=transform_strong)
    # train_unlabeled_dataset = iNatDataset(alg, data_dir, ulb_split, dataset, is_ulb=True, transform=transform_weak, transform_medium=transform_medium, transform_strong=transform_strong)
    # test_dataset = iNatDataset(alg, data_dir, 'test', dataset, transform=transform_val)

    l_train_dir = os.path.join("/home/lhz/data/semi-inat", 'train/l_train')
    id_train_dir = os.path.join("/home/lhz/data/semi-inat", 'train/u_train/id')
    test_dir = os.path.join("/home/lhz/data/semi-inat", 'test')

    l_train_dset = datasets.ImageFolder(l_train_dir)
    id_train_dset = datasets.ImageFolder(id_train_dir)
    test_dset = datasets.ImageFolder(test_dir)

    l_train_data, l_train_targets = split_images_labels(l_train_dset.imgs)
    id_train_data, id_train_targets = split_images_labels(id_train_dset.imgs)
    test_data, test_targets = split_images_labels(test_dset.imgs)
    train_data = np.concatenate([l_train_data, id_train_data])
    train_targets = np.concatenate([l_train_targets, id_train_targets])

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
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    # if alg == 'fullysupervised':
    #     lb_data = train_data
    #     lb_targets = train_targets

    train_labeled_dataset = iNatDataset(alg, data_dir, train_split, dataset, transform=transform_weak,
                                        transform_medium=transform_medium, transform_strong=transform_strong,
                                        samples=lb_data, targets=lb_targets, num_classes=num_classes)
    train_unlabeled_dataset = iNatDataset(alg, data_dir, ulb_split, dataset, is_ulb=True, transform=transform_weak,
                                          transform_medium=transform_medium, transform_strong=transform_strong,
                                          samples=ulb_data, targets=ulb_targets, num_classes=num_classes)
    test_dataset = iNatDataset(alg, data_dir, 'test', dataset, transform=transform_val, samples=test_data,
                               targets=test_targets, num_classes=num_classes)

    num_data_per_cls = [0] * train_labeled_dataset.num_classes
    for l in train_labeled_dataset.targets:
        num_data_per_cls[l] += 1
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def make_dataset(dataset_root, split, task='All', pl_list=None):
    split_file_path = os.path.join(dataset_root, task, split + '.txt')

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, task, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    num_classes = len(set(classes))
    print('# images in {}: {}'.format(split, len(img)))
    return img, num_classes, classes


class iNatDataset(BasicDataset):
    def __init__(self, alg, dataset_root, split, task='All', transform=None, transform_medium=None,
                 transform_strong=None, loader=dataset_parser.default_loader, pl_list=None, is_ulb=False,
                 samples=None, targets=None, num_classes=None):
        super().__init__(alg, samples, targets=targets, is_ulb=is_ulb, transform=transform,
                         medium_transform=transform_medium, strong_transform=transform_strong, num_classes=num_classes)

        # self.alg = alg
        # self.is_ulb = is_ulb
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        # self.samples, self.num_classes, self.targets = make_dataset(self.dataset_root, split, self.task, pl_list=pl_list)
        self.samples, self.num_classes, self.targets = samples, num_classes, targets

        # self.transform = transform
        # self.medium_transform = transform_medium
        # if self.medium_transform is None:
        #     if self.is_ulb:
        #         assert self.alg not in ['sequencematch'], f"alg {self.alg} requires medium augmentation"
        #
        # self.strong_transform = transform_strong
        # if self.strong_transform is None:
        #     if self.is_ulb:
        #         assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher',
        #                                 'mixmatch', 'refixmatch', 'semipt'], f"alg {self.alg} requires strong augmentation"

        self.data = []
        for i in range(len(self.samples)):
            # self.data.append(self.samples[i][0])
            self.data.append(self.samples[i])

    def __sample__(self, idx):
        # path, target = self.samples[idx]
        path = self.samples[idx]
        img = self.loader(path)
        target = self.targets[idx]
        return img, target

    def __len__(self):
        return len(self.data)
