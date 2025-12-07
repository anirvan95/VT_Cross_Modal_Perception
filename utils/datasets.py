import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Dataset specific parameters
vis_obs_dim = [64, 64, 2]
tac_obs_dim = [80, 80, 1]
action_dim = 9
horizon = 99


class CrossModal(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
           file_paths (list of str): List of file paths to load the data.
        """
        self.file_paths = file_paths
        self.scale_vis = False
        self.all_vis_obs, self.all_tac_obs, self.all_actions, self.all_gt_obs = self._load_data_into_ram()

    def _load_data_into_ram(self):
        """Load all npz files into RAM at initialization."""
        all_actions = []
        all_vis_obs = []
        all_tac_obs = []
        all_gt_obs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for file_path in self.file_paths:
            with np.load(file_path) as data:
                if self.scale_vis:
                    vis_obs = data['vis_obs'] # Shape: (N, H, 128, 128, 2)
                    num_int = vis_obs.shape[0]
                    vis_obs = torch.tensor(vis_obs, dtype=torch.float16).to(device)
                    vis_obs = vis_obs.reshape(-1, 128, 128, 2)
                    vis_obs = vis_obs.permute(0, 3, 1, 2)  # Channel first
                    vis_obs = F.interpolate(vis_obs, size=(64, 64), mode='bilinear', align_corners=False)  # Rescale
                    vis_obs = vis_obs.permute(0, 2, 3, 1)  # Change back
                    vis_obs = vis_obs.reshape(num_int, horizon, 64, 64, 2)
                    all_vis_obs.append(vis_obs.to("cpu").numpy())
                else:
                    all_vis_obs.append(data['vis_obs'])
                all_tac_obs.append(data['tac_obs'])
                all_actions.append(data['action'])
                all_gt_obs.append(data['gt_obs'])

        # Convert lists of arrays into single numpy arrays
        all_vis_obs = np.concatenate(all_vis_obs, axis=0)
        all_tac_obs = np.concatenate(all_tac_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_gt_obs = np.concatenate(all_gt_obs, axis=0)

        return (torch.tensor(all_vis_obs, dtype=torch.float16),
                torch.tensor(all_tac_obs, dtype=torch.float16),
                torch.tensor(all_actions, dtype=torch.float16),
                torch.tensor(all_gt_obs, dtype=torch.float16))

    def __len__(self):
        return self.all_actions.shape[0]

    def __getitem__(self, idx):
        vis_obs = self.all_vis_obs[idx]
        tac_obs = self.all_tac_obs[idx]
        tac_obs = tac_obs.unsqueeze(-1)
        action = self.all_actions[idx]
        gt_obs = self.all_gt_obs[idx]

        return vis_obs, tac_obs, action, gt_obs
