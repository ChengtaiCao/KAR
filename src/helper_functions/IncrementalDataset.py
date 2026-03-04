"""
IncrementalDataset.py
"""

import os
import torch
from .coco_loader import COCOLoader
from .voc_loader import VOC


def build_dataset(dataset_name, root_dir, low_range, high_range, phase=None, year='2014', transform=None):
    """
    Get Training and Validation Datasets for the Given Range
    """

    # Dataset File Name
    assert phase in ['train', 'val'], "phase should be \'train\' or \'val\'"
    dataset_name = dataset_name.lower()
    file_name = phase + year

    # Included Classes
    retrieve_classes = range(low_range, high_range)
    # Load Dataset
    if 'coco' in dataset_name:
        instances_path = os.path.join(root_dir, f'annotations/instances_{file_name}.json')
        data_path = f'{root_dir}/{file_name}'

        dataset = COCOLoader(data_path, instances_path, included=retrieve_classes,
                             transform=transform)

    elif dataset_name == 'voc2007':
        if phase == 'train':
            dataset = VOC('07', 'edgeboxes', 'trainval', included=retrieve_classes, root=root_dir,
                          transform=transform)
        if phase == 'val':
            dataset = VOC('07', 'edgeboxes', 'test', included=retrieve_classes, root=root_dir,
                          transform=transform)
    return dataset


def build_loader(dataset, batch_size, num_workers, phase=None):
    assert phase in ['train', 'val'], "phase should be \'train\' or \'val\'"
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size,
                                             num_workers=num_workers, 
                                             pin_memory=True,
                                             shuffle=(phase == 'train')
                                             )
    return dataloader


class ModifiableSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.modifications = {}

    def __getitem__(self, idx):
        if idx in self.modifications:
            return self.modifications[idx]
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

    def __len__(self):
        return len(self.indices)
    
    def modify_items_batch(self, batch_indices, threshold_buffer_mask, low_range, high_range):
        for i, idx in enumerate(batch_indices):
            original_idx = self.indices[idx]
            img, target = self.dataset[original_idx]
            target[low_range:high_range][threshold_buffer_mask[i]] = 1
            self.modifications[idx] = (img, target)
    