# TODO: add baseline model to compare the cross-modal implementation
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.dist as dist
from utils.functions import bayes_fusion

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Dataset specific parameters
vis_obs_dim = [128, 128, 2]
tac_obs_dim = [80, 80, 1]
action_dim = 9
horizon = 99

from networks import *

class MultiModalLF(nn.Module):
    def __init__(self, args):
        super(MultiModalLF, self).__init__()
        self.vis_dim_z = args.vis_dim_z
        self.tac_dim_z = args.tac_dim_z

    def multimodal_filter(self):
        # TODO: Add the multi modal filtering step.
        print('Baseline multi_modal_filter implementation in progress, no with merged space')