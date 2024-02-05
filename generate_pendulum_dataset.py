import time
import gym as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper
from wrapper import PixelDictWrapper, PendulumEnv
import itertools


def collect_data(num_sequences: int, sequence_length: int):
    data = dict(obs=[], actions=[], states=[], parameters=[])
    # Generate num_sequences trajectories for each parameters
    l = np.linspace(0.5, 1.5, 5)    # modify the variation in length of pendulum
    m = np.linspace(0.1, 1.0, 5)    # modify the variation in mass of pendulum
    mu = np.linspace(0.1, 0.5, 5)  # modify the variation in joint friction of pendulum

    for params in itertools.product(l, m, mu):
        env = PendulumEnv(params)
        env.reset()
        env = PixelDictWrapper(PixelObservationWrapper(env))
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=(32, 32))
        episodes = 0
        while episodes < num_sequences:
            print("Episode number: ", episodes, "Current parameters:", params)
            t = 0
            vis_obs = env.reset()
            vis_obs = 255 - vis_obs
            state = np.array([env.state[0], env.state[1]], dtype=np.float32)
            state = state[:, None]
            observations, actions, states, parameters = [], [], [], []
            while t < sequence_length:
                action = env.action_space.sample()
                observations.append(vis_obs)
                actions.append(action)
                states.append(state)
                parameters.append(params)
                vis_obs, state, _, _ = env.step(action)
                vis_obs = 255 - vis_obs
                # cv2.imshow(mat=vis_obs, winname='generated')
                # cv2.waitKey(5)
                t += 1

            data['obs'].append(observations)
            data['actions'].append(actions)
            data['states'].append(states)
            data['parameters'].append(parameters)
            episodes += 1

        env.close()
        del env

        np.savez(f'datasets/pendulum/training_params.npz', obs=np.array(data['obs']), actions=np.asarray(data['actions']), states=np.array(data['states']), parameters=np.array(data['parameters']))


if __name__ == '__main__':
    collect_data(1000, 15)