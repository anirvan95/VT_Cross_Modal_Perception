import numpy as np
import matplotlib.pyplot as plt

filename = 'adc_data_Raw_28_tarak_0.5_normal_breathing_Raw_0.bin'
num_frames = 4000
num_rx = 4
num_samples = 256

# Read the binary file as int16 (2-byte values)
raw_data = np.fromfile(filename, dtype=np.int16)
file_size = raw_data.shape[0]
num_chirps = file_size // (2 * num_samples * num_rx)
# Combine real and imaginary parts into complex data
lvds = np.zeros([1, file_size // 2], dtype=np.complex64)
counter = 0
for i in range(0, file_size - 1, 4):
    lvds[0, counter] = raw_data[i] + 1j * raw_data[i + 2]
    lvds[0, counter + 1] = raw_data[i + 1] + 1j * raw_data[i + 3]
    counter += 2

# lvds[0, :] = raw_data[::2] + 1j * raw_data[1::2]
lvds = lvds.reshape((num_samples*num_rx, num_chirps), order='F')
lvds = lvds.T

print('Done processing')

# Organize data into receiver channels
adc_data_reshaped = np.zeros((num_rx, num_chirps * num_samples), dtype=np.complex64)
for row in range(num_rx):
    for i in range(num_chirps):
        start_idx = (i * num_samples)
        end_idx = (i + 1) * num_samples
        adc_data_reshaped[row, start_idx:end_idx] = lvds[i, row * num_samples:(row + 1) * num_samples]

print(abs(adc_data_reshaped[0, 0:10]))
print('test')
