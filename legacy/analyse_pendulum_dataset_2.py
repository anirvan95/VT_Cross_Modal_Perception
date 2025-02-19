import numpy as np
import torch
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

dim_x = (32, 32)
dim_u = 1
dim_p = 3
dim_s = 2
batch_size = 1


def load_data(file: str, device='cpu') -> Dataset:
    x = torch.from_numpy(np.load(file)['obs']).to(device)
    x = x.to(torch.float32) / 255
    x = x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
    u = torch.from_numpy(np.load(file)['actions']).to(device)
    u = u.to(torch.float32)
    p = torch.from_numpy(np.load(file)['parameters']).to(device)
    p = p.to(torch.float32)
    s = torch.from_numpy(np.load(file)['states']).to(device)
    s = s.to(torch.float32)

    return TensorDataset(x, u, s, p)


device = 'cpu'
datasets = dict((k, load_data(file=f'datasets/pendulum/{k}.npz', device=device)) for k in ['training_params'])
train_loader = DataLoader(datasets['training_params'], batch_size=batch_size, shuffle=True)

i = 0
for batch in train_loader:
    x, u, s, p = batch[0], batch[1], batch[2], batch[3]
    print(i)
    i += 1
