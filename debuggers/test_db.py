import pickle
import matplotlib.pyplot as plt
import numpy as np
import imageio
import matplotlib.cm as cm

def normalize_tactile_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = np.asarray(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val + 1e-8) # Avoid division by zero

def apply_colormap(image, cmap='viridis'):
    colormap = cm.get_cmap(cmap)  # Get the colormap
    colored_image = colormap(image)[:, :, :3]*127.5 + 127.5  # Apply colormap and drop alpha channel
    colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
    return colored_image

with open('dataset/cm_dataset/preprocessed/c-1-1.pickle', 'rb') as handle:
    db = pickle.load(handle)

tt = db[0]
tac = tt['tactile_obs']
'''
print(tac.shape)
fig, ax = plt.subplots(5, 5)
count = 0
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(tac[count, :, :])
        count += int(len(tac)/25)
plt.show()
'''
tactile_colored_frames = [apply_colormap(frame, 'viridis') for frame in tac]
imageio.mimsave('dump/tactile/test.gif', tactile_colored_frames, fps=9)
