# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:39:54 2024

@author: simon
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Define the neural network
class GaussianMapper(nn.Module):
    def __init__(self):
        super(GaussianMapper, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# # Transform gaussian varaibles. Mimick the non-linear correlation between visual and haptic latent spaces
# def mean_transform(variable):
#     noise = np.random.randn(*variable.shape)/10
#     transformed_variable = (np.sin(variable) + np.cos(variable**2) - np.exp(-variable) + 
#                        np.log(np.abs(variable) + 1) ** 2 + np.tan(variable / 3.0)) + noise
#     return transformed_variable

# # Transform gaussian varaibles. Mimick the non-linear correlation between visual and haptic latent spaces
# def std_transform(variable):
#     noise = np.random.randn(*variable.shape)/5
#     transformed_variable = (np.cos(variable) + np.sin(variable**2) + np.exp(-variable/3) - 
#                        np.log(np.abs(variable) + 1) ** 3 + np.tan(variable / 2.0)) + noise
#     return transformed_variable

# Transform gaussian varaibles. Mimick the non-linear correlation between visual and haptic latent spaces
def complex_transform(variable):
    noise = np.random.randn(*variable.shape)/10
    transformed_variable = (np.sin(variable) + np.cos(variable**2) - np.exp(-variable) + 
                       np.log(np.abs(variable) + 1) ** 2 + np.tan(variable / 3.0)) + noise
    return transformed_variable

# Initialize the network, loss function, and optimizer
net = GaussianMapper()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Generate training data
mu1, sigma1 = np.random.randn(2, 32, 100)
mu2 = complex_transform(mu1)
sigma2 = complex_transform(sigma1)
source_gaussians = np.concatenate((mu1, sigma1), axis=0).astype(np.float32)
source_gaussians = torch.from_numpy(source_gaussians).view(-1, 1)
target_gaussians = np.concatenate((mu2, sigma2), axis=0).astype(np.float32)
target_gaussians = torch.from_numpy(target_gaussians).view(-1, 1)

train_data = TensorDataset(source_gaussians, target_gaussians)
batch_size = 100

# Split your data into a training set and a validation set
train_data, test_data = torch.utils.data.random_split(train_data, 
                     [int(0.8*len(train_data)), len(train_data)-int(0.8*len(train_data))])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)



# In your training loop, after updating your model, evaluate it on the validation set
for epoch in range(1000):
    # Training phase
    for batch in train_loader:
        source_batch, target_batch = batch
        optimizer.zero_grad()
        outputs = net(source_batch)
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()

    # Validation phase
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            source_batch, target_batch = batch
            outputs = net(source_batch)
            loss = criterion(outputs, target_batch)
            test_loss += loss.item()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {test_loss/len(test_loader)}')
