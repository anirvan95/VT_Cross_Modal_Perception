# Generates the output.npz for a particular checkpoint and iterations
import os
import re
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from utils.baseline_svae import SVAE
from utils.datasets import CrossModal
import matplotlib

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def apply_colormap(image, cmap='viridis'):
    colormap = matplotlib.colormaps[cmap]  # Get the colormap
    colored_image = colormap(image)[:, :, :3]*255  # Apply colormap and drop alpha channel
    colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
    return colored_image


def parse_value(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    try:
        if "." in value or "e" in value.lower():  # Handle floats including scientific notation
            return float(value)
        return int(value)  # Convert to int if possible
    except ValueError:
        return value  # Return as string if conversion fails


def load_args_from_file(file_path):
    args = argparse.Namespace()
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ", 1)
            setattr(args, key, parse_value(value))

    return args


def main():
    model_path = os.path.join('results', 'baseline')
    args = load_args_from_file(os.path.join(model_path, 'args.txt'))
    svae = SVAE(args)

    # Data Loader
    print('Loading Dataset')
    test_dir = os.path.join('dataset', 'cm_dataset', 'test_set')
    test_file_paths = sorted(
        [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.npz')],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())  # Extract first numeric part
    )
    test_dataset = CrossModal(test_file_paths)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    # Load the trained model
    checkpoints_path = os.path.join(model_path, 'latent_model.pth')
    checkpt = torch.load(checkpoints_path, weights_only=False)
    state_dict = checkpt['state_dict']
    svae.load_state_dict(state_dict)
    print('Latent model loaded successfully')

    svae.eval()

    first = True
    noise = 0  # std deviation of the noise to be passed
    corruption = 0  # ratio of time horizon data is corrupted

    with torch.no_grad():
        for values in test_loader:
            vis_obs, tac_obs, actions, labels = values
            vis_obs = vis_obs.cuda().to(dtype=torch.float32)
            tac_obs = tac_obs.cuda().to(dtype=torch.float32)
            if noise > 0:
                tac_obs = tac_obs + torch.normal(0, noise, size=tac_obs.shape).cuda()
                vis_obs = vis_obs + torch.normal(0, noise, size=vis_obs.shape).cuda()
            elif corruption > 0:
                mask = torch.ones_like(vis_obs)
                for b in range(vis_obs.shape[0]):
                    idx = torch.randperm(vis_obs.shape[1])[:int(corruption * vis_obs.shape[1])]
                    mask[b, idx] = 0
                vis_obs = vis_obs * mask

                mask = torch.ones_like(tac_obs)
                for b in range(tac_obs.shape[0]):
                    idx = torch.randperm(tac_obs.shape[1])[:int(corruption * tac_obs.shape[1])]
                    mask[b, idx] = 0
                tac_obs = tac_obs * mask

            actions = actions.cuda().to(dtype=torch.float32)
            labels = labels.cuda().to(dtype=torch.float32)

            object_labels = labels[:, :, 6:]  # Required for hierarchical prior, contrastive version is more general
            pose_labels = labels[:, :, :6]  # GT Pose of the object

            (prior_params_y_f, y_params_f, ys_f, hs_f, cs_f,
             vis_prior_params_z_f, vis_z_params_f, vis_zs_f, vis_xs_hat_f, vis_x_hat_params_f, vis_x_f,
             tac_prior_params_z_f, tac_z_params_f, tac_zs_f, tac_xs_hat_f, tac_x_hat_params_f,
             tac_x_f) = svae.filter(vis_obs, tac_obs, actions, object_labels)

            # Move results to CPU to reduce GPU memory usage
            y_params_f = torch.stack(y_params_f, dim=1).cpu()
            vis_z_params_f = torch.stack(vis_z_params_f, dim=1).cpu()
            tac_z_params_f = torch.stack(tac_z_params_f, dim=1).cpu()

            if first:
                y_params = y_params_f
                vis_z_params = vis_z_params_f
                tac_z_params = tac_z_params_f
                labels_test = object_labels.cpu()
                pose_test = pose_labels.cpu()
                first = False
            else:
                y_params = torch.cat((y_params, y_params_f), dim=0)
                vis_z_params = torch.cat((vis_z_params, vis_z_params_f), dim=0)
                tac_z_params = torch.cat((tac_z_params, tac_z_params_f), dim=0)
                labels_test = torch.cat((labels_test, object_labels.cpu()), dim=0)
                pose_test = torch.cat([pose_test, pose_labels.cpu()], dim=0)

    print('Done processing the dataset')

    # Save the variables.
    y_params = y_params.numpy()
    vis_z_params = vis_z_params.numpy()
    tac_z_params = tac_z_params.numpy()
    labels_test = labels_test.numpy()
    pose_test = pose_test.numpy()

    np.savez(os.path.join(model_path, 'output_test' + '_n'+str(noise)+'_c'+str(corruption)+'.npz'),
             y_params=y_params,
             vis_z_params=vis_z_params,
             tac_z_params=tac_z_params,
             labels_test=labels_test,
             pose_test=pose_test)

if __name__ == '__main__':
    evaluator = main()