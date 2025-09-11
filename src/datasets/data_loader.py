import os
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import SVHN as TorchVisionSVHN

from . import base_dataset as basedat
from . import memory_dataset as memd
from .dataset_config import dataset_config

from networks.cipnet_utils.util.data import TrivialAugmentWideNoColor, TrivialAugmentWideNoShape, TrivialAugmentWideNoShapeWithColor
from torch.utils.data.distributed import DistributedSampler


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers, pin_memory, validation=.1,
                repeat_task_0=False, parallelization="DP"):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load, psh_load = [], [], [], []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        print(f"Processing dataset: {cur_dataset}")
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]
        print(dc)
        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'],
                                                      online_augment=dc['online_augment'],)
        # datasets
        if 'cub_200_2011' in cur_dataset or 'cars' in cur_dataset:
            img_size = 224 # SPERO SIA 224 SEMPRE
            shape = (3, img_size, img_size)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            normalize = transforms.Normalize(mean=mean,std=std)
            
            transform_noaug = transforms.Compose([
                                    transforms.Resize(size=(img_size, img_size)),
                                    transforms.ToTensor(),
                                    normalize
                                ])
            # transform1p = None
            
            transform1 = transforms.Compose([
                transforms.Resize(size=(img_size+8, img_size+8)), 
                TrivialAugmentWideNoColor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
            ])
            notran = transforms.Compose([
                transforms.Resize(size=(img_size+8, img_size+8)), 
            ])
            # transform1p = transforms.Compose([
            #     transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            #     TrivialAugmentWideNoColor(),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
            # ])
            
            transform2 = transforms.Compose([
                                TrivialAugmentWideNoShape(),
                                transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                                transforms.ToTensor(),
                                normalize
                                ])
            
            print("VALIDATION",validation)
            trn_dset, val_dset, tst_dset, psh_dset, curtaskcla, d_idxs = get_cub_datasets(cur_dataset, dc['path'], num_tasks,
                                                                                nc_first_task,
                                                                                validation=validation,
                                                                                trn_transform=trn_transform,
                                                                                tst_transform=tst_transform,
                                                                                cipnet=True,
                                                                                transform1=transform1,
                                                                            #  transform1=notran,################
                                                                                transform2=transform2,
                                                                                transform_noaug=transform_noaug,
                                                                                class_order=dc['class_order'],
                                                                                repeat_task_0=repeat_task_0)
            
        
        else:
            trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                    validation=validation,
                                                                    trn_transform=trn_transform,
                                                                    tst_transform=tst_transform,
                                                                    class_order=dc['class_order'])

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
                psh_dset[tt].labels = [elem + dataset_offset for elem in psh_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            print(f"Task {tt}: Training dataset size = {len(trn_dset[tt])}")
            print(f"Task {tt}: Validation dataset size = {len(val_dset[tt])}")
            print(f"Task {tt}: Test dataset size = {len(tst_dset[tt])}")
            if parallelization=="DDP":
                trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory, sampler=DistributedSampler(trn_dset[tt])))
                val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory, sampler=DistributedSampler(val_dset[tt])))
                tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory, sampler=DistributedSampler(tst_dset[tt])))
                psh_load.append(data.DataLoader(psh_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory))
            else:
                trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                pin_memory=pin_memory))
                val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory))
                tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory))
                psh_load.append(data.DataLoader(psh_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=pin_memory))
    return trn_load, val_load, tst_load, psh_load, taskcla, d_idxs


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, class_order=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []

    if 'mnist' in dataset:
        tvmnist_trn = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_tst = TorchVisionMNIST(path, train=False, download=True)
        trn_data = {'x': tvmnist_trn.data.numpy(), 'y': tvmnist_trn.targets.tolist()}
        tst_data = {'x': tvmnist_tst.data.numpy(), 'y': tvmnist_tst.targets.tolist()}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        trn_data = {'x': tvsvhn_trn.data.transpose(0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        tst_data = {'x': tvsvhn_tst.data.transpose(0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # Notice that SVHN in Torchvision has an extra training set in case needed
        # tvsvhn_xtr = TorchVisionSVHN(path, split='extra', download=True)
        # xtr_data = {'x': tvsvhn_xtr.data.transpose(0, 2, 3, 1), 'y': tvsvhn_xtr.labels}

        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'imagenet_32' in dataset:
        import pickle
        # load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # labels from 0 to 999
        # reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        trn_data = {'x': x_trn, 'y': y_trn}
        tst_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    else:
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                            validation=validation, shuffle_classes=class_order is None,
                                                            class_order=class_order)
        # set dataset type
        Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices, name=dataset))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices, name=dataset))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices, name=dataset))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla


def get_cub_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, cipnet=False, transform1=None, transform2=None, transform_noaug=None, class_order=None, repeat_task_0=False):
    """Extract datasets and create Dataset class"""
    # print("nc_first_task",nc_first_task, flush=True)
    # print("class_order",class_order, flush=True)
    trn_dset, val_dset, tst_dset, psh_dset = [], [], [], []

    # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
    # all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
    #                                                     validation=validation, shuffle_classes=class_order is None,
    #                                                     class_order=class_order)
    all_data, taskcla, class_indices, d_idxs = basedat.get_data2(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                        validation=validation, shuffle_classes=class_order is None,
                                                        class_order=class_order)
    # set dataset type
    Dataset = basedat.BaseDataset
    if cipnet:
        # print("scelto basedatasetcipnet")
        Dataset=basedat.BaseDatasetCIPNet

    if repeat_task_0:
        # print("SONO ENTRATO IN REPEAT TASK 0")
        taskcla.insert(0, taskcla[0])
    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['psh']['y'] = [label + offset for label in all_data[task]['psh']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        if cipnet:
            trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices, name=dataset,
                                    transform1=transform1, transform2=transform2))
            val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices, name=dataset,
                                   transform1=transform1, transform2=transform2, transform_noaug=transform_noaug, test=True))
            tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices, name=dataset,
                                   transform1=transform1, transform2=transform2, transform_noaug=transform_noaug, test=True))
            psh_dset.append(Dataset(all_data[task]['psh'], tst_transform, class_indices, name=dataset,
                                   transform1=transform1, transform2=transform2))
        else:
            trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices, name=dataset))
            val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices, name=dataset))
            tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices, name=dataset))
            psh_dset.append(Dataset(all_data[task]['psh'], tst_transform, class_indices, name=dataset))
        
        offset += taskcla[task][1]
    if repeat_task_0:
        # print("SONO ENTRATO IN REPEAT TASK 0")
        if cipnet:
            trn_dset.insert(0, Dataset(all_data[0]['trn'], trn_transform, class_indices, name=dataset,
                                      transform1=transform1, transform2=transform2))
            val_dset.insert(0, Dataset(all_data[0]['val'], trn_transform, class_indices, name=dataset,
                                      transform1=transform1, transform2=transform2, transform_noaug=transform_noaug, test=True))
            tst_dset.insert(0, Dataset(all_data[0]['tst'], trn_transform, class_indices, name=dataset,
                                      transform1=transform1, transform2=transform2, transform_noaug=transform_noaug, test=True))
            psh_dset.insert(0, Dataset(all_data[0]['psh'], trn_transform, class_indices, name=dataset,
                                      transform1=transform1, transform2=transform2))
        else:
            trn_dset.insert(0, Dataset(all_data[0]['trn'], trn_transform, class_indices, name=dataset))
            val_dset.insert(0, Dataset(all_data[0]['val'], trn_transform, class_indices, name=dataset))
            tst_dset.insert(0, Dataset(all_data[0]['tst'], trn_transform, class_indices, name=dataset))
            psh_dset.insert(0, Dataset(all_data[0]['psh'], trn_transform, class_indices, name=dataset))

    return trn_dset, val_dset, tst_dset, psh_dset, taskcla, d_idxs


def get_transforms(resize, pad, crop, flip, normalize, extend_channel, online_augment):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    if online_augment:
        online_augmentation = transforms.Compose([
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
            ]),
        ])
        trn_transform_list.append(online_augmentation)

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)

