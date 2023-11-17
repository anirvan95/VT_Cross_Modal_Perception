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
dim_a = 16
dim_w = 3


def make_env():
    env = PendulumEnv()
    env.reset()
    env = PixelDictWrapper(PixelObservationWrapper(env))
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(16, 16))
    #print(env.action_space)
    #print(env.observation_space)
    return env


def collect_data(num_sequences: int, sequence_length: int):
    data = dict(obs=[], actions=[])
    env = make_env()
    episodes = 0
    while episodes < num_sequences:
        print("Episode number: ", episodes)
        obs = env.reset()
        done, t = False, 0
        observations, actions = [], []
        while t <= sequence_length:
            action = env.action_space.sample()
            observations.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            cv2.imshow(mat=obs, winname='generated')
            cv2.waitKey(5)
            #plt.imshow(obs)
            #plt.show()
            t += 1
            if t == sequence_length:
                data['obs'].append(observations)
                data['actions'].append(actions)
                observations, actions = [], []
                episodes += 1
    np.savez(f'datasets/pendulum/validation_mod.npz', obs=np.array(data['obs']), actions=np.asarray(data['actions']))


if __name__ == '__main__':
    collect_data(500, 20)