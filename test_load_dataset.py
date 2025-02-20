import torch
from utils.datasets import CrossModal
from torch.utils.data import DataLoader
import numpy as np
import os
import time


train_dir = 'dataset/cm_dataset/training'
file_paths = sorted([os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.npz')])

start = time.time()
dataset = CrossModal(file_paths)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
train_loader = DataLoader(dataset, batch_size=3, shuffle=True) # Debug mode

for i, values in enumerate(train_loader):
    vis_obs, tac_obs, actions, gt_labels = values
    # print(vis_obs.shape)


print('Time taken - ', time.time() - start)
