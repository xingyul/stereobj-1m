


from torch.utils.data.dataloader import default_collate
import time
import torch
import numpy as np
import torch.utils.data


from transforms import make_transforms
import stereobj1m_dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, batch_size, drop_last):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(args, lr=False, split='train'):
    if 'train' in split:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    is_train = 'train' in split

    transforms = make_transforms(is_train=is_train)
    dataset = stereobj1m_dataset.Dataset(args, lr=lr, transforms=transforms)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, args.batch_size, drop_last)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=default_collate,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    return data_loader



