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
import itertools


def collect_data(num_sequences: int, sequence_length: int):
    data = dict(obs=[], actions=[], states=[])
    # Generate num_sequences trajectories for each parameters
    l = np.linspace(0.5, 1.5, 5)
    m = np.linspace(0.1, 1.0, 5)
    mu = np.linspace(0.01, 0.5, 5)

    for parameters in itertools.product(l, m, mu):
        env = PendulumEnv(parameters)
        env.reset()
        env = PixelDictWrapper(PixelObservationWrapper(env))
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=(32, 32))
        observations, actions, states = [], [], []
        episodes = 0
        while episodes < num_sequences:
            print("Episode number: ", episodes)
            print("Current parameters:", parameters)
            t = 0
            vis_obs = env.reset()
            vis_obs = 255 - vis_obs
            state = env.state
            while t <= sequence_length:
                action = env.action_space.sample()
                observations.append(vis_obs)
                actions.append(action)
                states.append(state)
                vis_obs, state, _, _ = env.step(action)
                vis_obs = 255 - vis_obs
                # cv2.imshow(mat=vis_obs, winname='generated')
                # cv2.waitKey(5)
                # plt.imshow(obs)
                # plt.show()
                t += 1
                if t == sequence_length:
                    data['obs'].append(observations)
                    data['actions'].append(actions)
                    data['states'].append(state)
                    observations, actions, states = [], [], []
                    episodes += 1

        env.close()
        del env


# np.savez(f'dataset/training_ext.npz', obs=np.array(data['obs']), actions=np.asarray(data['actions']))


if __name__ == '__main__':
    collect_data(10, 15)