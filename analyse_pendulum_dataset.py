import time
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
'''
file = 'dataset/training.npz'
input_data = data['data']
fig, ax = plt.subplots(1, 15)
for i in range(0, 15):
    image_seq_example = input_data[100, i, 0:256]
    im = np.reshape(image_seq_example, [16, 16])
    ax[i].imshow(im, origin='lower')

plt.show()
'''
dim_z = 3
dim_x = (16, 16)
dim_u = 1
dim_a = 16
dim_w = 3
batch_size = 32
num_iterations = int(5000)
learning_rate = 0.01


def load_data(file: str, device='cpu') -> Dataset:
    data = torch.from_numpy(np.load(file)['data']).to(device)
    x, u = data[..., :-1], data[..., -1:]
    x = x.to(torch.float32) / 255. - 0.5
    return TensorDataset(x, u)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = dict((k, load_data(file=f'dataset/{k}.npz', device=device)) for k in ['training', 'test', 'validation'])

train_loader = DataLoader(datasets['training'], batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False)

i = 0
for batch in train_loader:
    x, u = batch[0], batch[1]
    print(u.shape)
    print(i)
    i += 1
