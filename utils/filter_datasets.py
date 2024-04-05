import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = '../datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        return x


class Pendulum(object):
    def __init__(self, dataset_zip=None):
        loc = 'datasets/pendulum/training_params.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc)
        else:
            self.dataset_zip = dataset_zip

        self.imgs = torch.from_numpy(self.dataset_zip['obs']).float()
        self.states = torch.from_numpy(self.dataset_zip['states']).float()
        self.params = torch.from_numpy(self.dataset_zip['parameters']).float()
        self.actions = torch.from_numpy(self.dataset_zip['actions']).float()
        self.imgs = self.imgs/255
        num_trajectories = self.imgs.shape[0]
        self.imgs = self.imgs.view(num_trajectories, 15, 32, 32)

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        obs = self.imgs[index].view(15, 32, 32)
        state = self.states[index].view(15, 2)
        parameter = self.params[index].view(15, 3)
        action = self.actions[index].view(15, 1)
        return obs, state, parameter, action


class Dataset(object):
    def __init__(self, loc):
        self.dataset = torch.load(loc).float().div(255).view(-1, 1, 64, 64)

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        return self.dataset[index]


class Faces(Dataset):
    LOC = '..datasets/basel_face_renders.pth'

    def __init__(self):
        return super(Faces, self).__init__(self.LOC)

