import torch
from utils.datasets import CrossModal
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import re

'''
def combine_dimensions(d1, d2, d3):
    unique_value = (d1 - 1) * (3 * 5) + (d2 - 1) * 5 + (d3 - 1)
    return unique_value

# Example usage:
for d1 in range(1, 6):
    for d2 in range(1, 4):
        for d3 in range(1, 6):
            print(f"({d1}, {d2}, {d3}) -> {combine_dimensions(d1, d2, d3)}")
'''

train_dir = 'dataset/cm_dataset/training'
val_dir = 'dataset/cm_dataset/validation'

# train_file_paths = sorted([os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.npz')])
val_file_paths = sorted(
    [os.path.join(val_dir, file) for file in os.listdir(val_dir) if file.endswith('.npz')],
    key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())  # Extract first numeric part
)

print(val_file_paths)
start = time.time()

# train_dataset = CrossModal(train_file_paths)
val_dataset = CrossModal(val_file_paths)

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True) # Debug mode
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for i, values in enumerate(test_loader):
    vis_obs, tac_obs, actions, gt_labels = values

# print(vis_obs.shape)

for i, values in enumerate(test_loader):
    vis_obs, tac_obs, actions, gt_labels = values
    print(gt_labels[:, 0, 6], gt_labels[:, 0, 7], gt_labels[:, 0, 8])
    print((gt_labels[:, 0, 6]-1)*15+(gt_labels[:, 0, 7]-1)*5+(gt_labels[:, 0, 8]-1))

print('Time taken - ', time.time() - start)
