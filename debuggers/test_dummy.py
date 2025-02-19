from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat("dataset/GDN0001/GDN0001_1_Resting.mat")

radar_i = data['radar_i']
radar_q = data['radar_q']

radar_i = radar_i[:, 0]
radar_q = radar_q[:, 0]

# plt.plot(radar_i[:, 0], radar_q[:, 0], '.r')
# plt.show()

# Form co-efficient matrix
M = np.column_stack([radar_i**2, radar_q**2, radar_i*radar_q, radar_i, radar_q])
v = np.ones(len(radar_i))

# Solve for ellipse co-efficient
x = np.linalg.svd(M, v, rcond=None)

# Calculate imbalances
A, B, C, D, E = x
amplitude_error = np.sqrt(1/A)
phase_error = np.arcsin(B/(2*np.sqrt(A)))
