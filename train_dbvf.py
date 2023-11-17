import time
import gym as gym
import imageio
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributions as td
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
from gym.wrappers import GrayScaleObservation, FlattenObservation, TransformObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper

from model import DVBF
from wrapper import PixelDictWrapper, PendulumEnv

dim_z = 3
dim_x = (16, 16)
dim_u = 1
dim_a = 1
dim_w = 3
batch_size = 120
num_iterations = int(1e5)
learning_rate = 0.01


def make_env():
    env = PendulumEnv()
    env.reset()
    env = PixelDictWrapper(PixelObservationWrapper(env))
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(16, 16))
    print(env.action_space)
    print(env.observation_space)
    return env

'''
def collect_data(num_sequences: int, sequence_length: int):
    data = dict(obs=[], actions=[])
    env = make_env()
    episodes = 0
    while episodes < num_sequences:
        obs = env.reset()
        done, t = False, 0
        observations, actions = [], []
        while not done and episodes < num_sequences:
            action = env.action_space.sample()
            observations.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            t += 1
            if t == sequence_length:
                t = 0
                data['obs'].append(observations)
                data['actions'].append(actions)
                observations, actions = [], []
                episodes += 1
    np.savez(f'dataset/raw.npz', obs=np.array(data['obs']), actions=np.asarray(data['actions']))
'''


def load_data(file: str, device='cpu') -> Dataset:
    x = torch.from_numpy(np.load(file)['obs']).to(device)
    x = x.to(torch.float32) / 255
    x = x.view([500, 20, 16*16])
    u = torch.from_numpy(np.load(file)['actions']).to(device)
    u = u.to(torch.float32) / 1.5
    return TensorDataset(x, u)


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    datasets = dict((k, load_data(file=f'dataset/{k}.npz', device=device)) for k in ['training_mod', 'validation_mod'])

    train_loader = DataLoader(datasets['training_mod'], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(datasets['validation_mod'], batch_size=batch_size, shuffle=False)

    dvbf = DVBF(dim_x=16*16, dim_u=dim_u, dim_z=dim_z, dim_w=dim_w).to(device)
    # dvbf = torch.load('dvbf.th').to(device)

    # optimizer = torch.optim.Adam(dvbf.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adadelta(dvbf.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    T = num_iterations
    count = 0
    for i in range(num_iterations):
        total_loss = 0
        dvbf.train(True)
        for batch in train_loader:
            if i < 5:
                dvbf.ci = 1e-2
            else:
                count += 1
                if i % 250 == 0:
                    dvbf.ci = np.minimum(1, (1e-2 + count / T))

            x, u = batch[0], batch[1]
            optimizer.zero_grad()
            loss = dvbf.loss(x, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # scheduler.step()
        writer.add_scalar('loss', scalar_value=total_loss, global_step=i)
        # writer.add_scalar('learning rate', scalar_value=scheduler.get_lr()[0], global_step=i)

        dvbf.train(False)
        total_val_loss = 0
        for batch in validation_loader:
            x, u = batch[0], batch[1]
            val_loss = dvbf.loss(x, u)
            total_val_loss += val_loss.item()
        writer.add_scalar('val_loss', scalar_value=total_val_loss, global_step=i)
        print(f'[Epoch {i}] train_loss: {total_loss}, val_loss: {total_val_loss}')
        if i % 500 == 0:
            torch.save(dvbf, 'checkpoints/dvbf.th')
            generate(filename=f'dvbf-epoch-{i}')

    torch.save(dvbf, 'dvbf.th')
    print("Model Saved")


def generate(filename):
    dvbf = torch.load('checkpoints/dvbf.th').to('cpu')
    dataset = load_data('dataset/validation_mod.npz')
    x = dataset[40][0].unsqueeze(dim=0)
    u = dataset[40][1].unsqueeze(dim=0)
    T = u.shape[1]
    z, _ = dvbf.filter(x=x[:1], u=u)
    reconstructed = dvbf.reconstruct(z).view(1, T, -1)

    def format(x):
        img = torch.clip((x + 0.5) * 255., 0, 255).to(torch.uint8)
        return img.view(-1, 16, 16).numpy()
    frames = []
    for i in range(T):
        gt = format(x[:, i])
        pred = format(reconstructed[:, i])
        img = np.concatenate([gt, pred], axis=1).squeeze()
        # cv2.imshow(mat=img, winname='generated')
        # cv2.waitKey(5)
        #plt.imshow(img)
        #plt.show()
        frames.append(img)
    with imageio.get_writer(f"{filename}.mp4", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)


if __name__ == '__main__':
    # collect_data(5, 15)
    train()
    # generate(filename='checkpoints/dvbf_500_new')