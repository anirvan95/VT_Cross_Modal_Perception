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

# ################################################ DL Model Definition ################################################
class VisEncoder(nn.Module):
    """
    CNN+FC visual encoder
    """
    def __init__(self, latent_dim, cnn_out_dim, device):
        super(VisEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.cnn_out_dim = cnn_out_dim

        # Define the CNN with 4 layers
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling to reduce dimensionality
        )

        # Fully connected layer to match the dimension
        self.fc = nn.Linear(self.cnn_out_dim, self.latent_dim)

        self.device = device

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = x.permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)
        z = self.fc(self.cnn_encoder(x).view(batch_size, -1)) # [B, latent_dim*2]
        return z



class VisDecoder(nn.Module):
    """
    FC + DCNN visual decoder
    """
    def __init__(self, latent_dim, cnn_out_dim, std, device):
        super(VisDecoder, self).__init__()
        self.device = device
        self.std = std
        self.cnn_out_dim = cnn_out_dim
        self.fc = nn.Linear(latent_dim, self.cnn_out_dim)

        # Define a 4-layer transposed CNN decoder
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # (4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # (8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4, padding=0),  # (32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=4, stride=4, padding=0),  # (128, 128)
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        # Pass through fully connected layers
        fc_out = self.fc(z)
        fc_out = fc_out.reshape(batch_size, 64, 2, 2)
        reconstructed = self.cnn_decoder(fc_out)
        mu_obs = reconstructed.permute(0, 2, 3, 1)
        sigma_obs = Variable(torch.ones(z.size(0), vis_obs_dim[0], vis_obs_dim[1], vis_obs_dim[2]) * np.log(self.std))
        sigma_obs = sigma_obs.to(self.device)
        obs_params = torch.cat([mu_obs.unsqueeze(-1), sigma_obs.unsqueeze(-1)], dim=-1)

        return obs_params


class TacEncoder(nn.Module):
    """
    CNN+FC tactile encoder
    """
    def __init__(self, latent_dim, cnn_out_dim, device):
        super(TacEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.cnn_out_dim = cnn_out_dim

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [80, 80] -> [80, 80]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [80, 80] -> [40, 40]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [40, 40] -> [20, 20]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [20, 20] -> [10, 10]
            nn.ReLU()
        )
        self.fc = nn.Linear(self.cnn_out_dim, self.latent_dim)
        self.device = device

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = x.permute(0, 3, 1, 2)
        z = self.fc(self.cnn_encoder(x).reshape(batch_size, -1))  # [batch_size, latent_dim*2]
        return z


class TacDecoder(nn.Module):
    """
    FC + DCNN tactile decoder
    """
    def __init__(self, latent_dim, cnn_out_dim, std, device):
        super(TacDecoder, self).__init__()
        self.device = device
        self.std = std
        self.cnn_out_dim = cnn_out_dim
        self.fc = nn.Linear(latent_dim, self.cnn_out_dim)

        # Define Transposed CNN decoder to reconstruct image
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [10,10] -> [20,20]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [20,20] -> [40,40]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [40,40] -> [80,80]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # Final output [80,80,1]
            nn.Sigmoid()  # Normalize to [0,1]
        )

    def forward(self, z):
        batch_size = z.shape[0]
        fc_out = self.fc(z)
        fc_out = fc_out.reshape(batch_size, 256, 10, 10)
        reconstructed = self.cnn_decoder(fc_out)
        mu_obs = reconstructed.permute(0, 2, 3, 1)
        sigma_obs = Variable(torch.ones(z.size(0), tac_obs_dim[0], tac_obs_dim[1], tac_obs_dim[2])*np.log(self.std))   # Check here
        sigma_obs = sigma_obs.to(self.device)
        obs_params = torch.cat([mu_obs.unsqueeze(-1), sigma_obs.unsqueeze(-1)], dim=-1)
        return obs_params


class MLPTransition(nn.Module):
    """
    MLP Transition function to model the non-linear dynamics
    """
    def __init__(self, input_dim, output_dim):
        super(MLPTransition, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)

        # Setup the non-linearity
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        z = h.view(x.size(0), self.output_dim)
        return z

class LSTMEncode(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMEncode, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights for input, forget, output, and cell gates
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x_t, hidden=None):
        h_t, c_t = hidden
        i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
        o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
        c_tilde_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
        c_t = f_t * c_t + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        x_t = self.fc(h_t)

        return x_t, (h_t, c_t)

# ###################################### Latent Filter Definition #####################################################
class VAE(nn.Module):
    def __init__(self, dim_z, std=0.75, modality='vision', use_cuda=True):
        super(VAE, self).__init__()

        self.dim_z = dim_z # Dimension of directly observable latent variable
        self.x_std = std # Standard deviation for the output reconstruction
        self.x_dist = dist.Normal()  # Reconstructed Distribution which is continuous

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = dist.Normal()
        self.q_dist = q_dist=dist.Normal()
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.dim_z, 2))
        self.use_cuda = use_cuda
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        if modality == 'vision':
            self.encoder = VisEncoder(latent_dim=dim_z * self.q_dist.nparams, cnn_out_dim=64*2*2, device=device) # CNN out dimension needs calculation
            self.decoder = VisDecoder(latent_dim=dim_z, std=self.x_std, cnn_out_dim=64*2*2, device=device) # CNN out dimension needs calculation

        elif modality == 'tactile':
            self.encoder = TacEncoder(latent_dim=dim_z * self.q_dist.nparams, cnn_out_dim=256*10*10, device=device)
            self.decoder = TacDecoder(latent_dim=dim_z, std=self.x_std, cnn_out_dim=256*10*10, device=device)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self.get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.dim_z, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params


class UniModalLF(nn.Module):
    def __init__(self, args):
        super(UniModalLF, self).__init__()
        self.dim_z = args.dim_z
        self.dim_y = args.dim_y
        self.dim_h = args.dim_h
        self.dim_a = action_dim

        self.x_std = args.stdx
        self.y_std = args.stdy

        self.modality = args.modality
        self.horizon = horizon

        # distribution family of p(y)
        self.prior_dist_y = dist.Normal()
        self.q_dist_y = dist.Normal()
        # hyperparameters for prior p(z) - Hierarchical prior or Normal prior
        # self.register_buffer('prior_params_y', torch.zeros(self.dim_y, 2))
        self.embedding = nn.Embedding(num_embeddings=args.num_objects, embedding_dim=self.dim_y)
        # Recognition Model  - Visual Encoder & Decoder
        self.vae = VAE(dim_z=self.dim_z, std=self.x_std, modality=self.modality, use_cuda=args.use_cuda)

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.y_net = LSTMEncode(input_size=self.dim_z + self.dim_a, hidden_size=self.dim_h, output_size=self.dim_y * self.q_dist_y.nparams)

        # Transition Network, computes TRANSITION (from previous visual state z, inferred y (time-invariant properties) and action)
        self.transition_net = MLPTransition(input_dim=self.dim_z + self.dim_y + self.dim_a, output_dim=self.dim_z * self.vae.q_dist.nparams)

        if args.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    def get_prior_params_y(self, labels):
        object_label = labels[:, 0, 0] + labels[:, 0, 1] + labels[:, 0, 2] # possible to add the interaction as well to further disentangle
        object_label = object_label.to(torch.device('cuda')).long() # TODO: Fix with device selection
        prior_params_mu = torch.tanh(self.embedding(object_label))
        prior_params_sigma = torch.ones(labels.size(0), self.dim_y, 1)*np.log(self.y_std)
        prior_params_sigma = prior_params_sigma.to('cuda')
        prior_params_y = torch.cat((prior_params_mu[:, :, None], prior_params_sigma), dim=-1)

        return prior_params_y

    def filter(self, x, u, labels, H):
        batch_size, T, _ = u.shape

        # Set the hidden layer of the LSTM to zero
        hidden_y_t = (torch.zeros(batch_size, self.dim_h, device='cuda'),
                      torch.zeros(batch_size, self.dim_h, device='cuda'))

        # Define the variables to store the filtering output
        prior_params_y_f = []
        y_params_f = []
        ys_f = []
        hs_f = []
        cs_f = []
        prior_params_z_f = []
        z_params_f = []
        zs_f = []
        x_hat_params_f = []
        xs_hat_f = []
        x_f = []

        # Convention
        # t = previous time step
        # t_1 = current time step

        # First time step is handled differently
        # Obtain the measurement latent space
        zs_meas_t, z_meas_params_t = self.vae.encode(x[:, 0])
        prior_params_z_t = self.vae.get_prior_params(batch_size)  # only for 1st time step

        zs_t = zs_meas_t
        z_params_t = z_meas_params_t

        xs_hat_t, x_hat_params_t = self.vae.decode(zs_t)

        prior_params_z_f.append(prior_params_z_t)
        z_params_f.append(z_params_t)
        zs_f.append(zs_t)
        xs_hat_f.append(xs_hat_t)
        x_hat_params_f.append(x_hat_params_t)
        x_f.append(x[:, 0])

        # Transfer the prior params z_t
        prior_params_y_t_1 = self.get_prior_params_y(labels)

        for t in range(1, T):  # Note the time index starts from 0 due to the correct control index
            u_t_1 = u[:, t]
            u_t = u[:, t - 1]
            x_t_1 = x[:, t]

            y_params_t_1, hidden_y_t_1 = self.y_net.forward(torch.cat([zs_t, u_t], dim=-1), hidden_y_t)
            y_params_t_1 = y_params_t_1.view(batch_size, self.dim_y, self.q_dist_y.nparams)

            # Sample the latent code y
            ys_t_1 = self.q_dist_y.sample(params=y_params_t_1)

            # Obtain the next step z's
            z_trans_params_t_1 = self.transition_net.forward(torch.cat([zs_t, u_t_1, ys_t_1], dim=-1))
            z_trans_params_t_1 = z_trans_params_t_1.view(batch_size, self.dim_z, self.vae.q_dist.nparams)

            zs_meas_t_1, z_meas_params_t_1 = self.vae.encode(x_t_1)
            # z_params_t_1 = z_trans_params_t_1 # without Bayes Integration
            z_params_t_1 = bayes_fusion(z_trans_params_t_1, z_meas_params_t_1)

            # Pass through the decoder
            zs_t_1 = self.vae.q_dist.sample(params=z_params_t_1)
            xs_hat_t_1, x_hat_params_t_1 = self.vae.decode(zs_t_1)

            # Save the parameters
            prior_params_y_f.append(prior_params_y_t_1)
            y_params_f.append(y_params_t_1)
            ys_f.append(ys_t_1)

            prior_params_z_f.append(prior_params_z_t)
            z_params_f.append(z_params_t_1)
            zs_f.append(zs_t_1)

            xs_hat_f.append(xs_hat_t_1)
            x_hat_params_f.append(x_hat_params_t_1)
            x_f.append(x_t_1)

            h, c = hidden_y_t
            hs_f.append(h)
            cs_f.append(c)

            hidden_y_t = hidden_y_t_1
            zs_t = zs_t_1
            prior_params_z_t = z_params_t_1

        return prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, hs_f, cs_f


class CrossModalLF(nn.Module):
    def __init__(self, args):
        super(CrossModalLF, self).__init__()
        self.vis_dim_z = args.vis_dim_z
        self.vis_lf = [] # create visual latent filter by calling UniModalLF()
        self.tac_lf = []

    def cross_modal_filter(self):
        # TODO: Add the cross modal filtering step by 24th Feb.
        print('cross_modal_filter implementation in progress')

class MultiModalLF(nn.Module):
    def __init__(self, args):
        super(MultiModalLF, self).__init__()
        self.vis_dim_z = args.vis_dim_z
        self.vis_lf = [] # create visual latent filter by calling UniModalLF()
        self.tac_lf = []

    def multimodal_filter(self):
        # TODO: Add the multi modal filtering step.
        print('Baseline multi_modal_filter implementation in progress, no with merged space')