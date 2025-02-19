import numpy as np
import torch
from torch.utils.data import Dataset

class CrossModalAdv(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
           file_paths (list of str): List of file paths to load the data.
        """
        self.file_paths = file_paths
        self.transform = transform

        # Store the lengths of data in each file to allow indexing across files
        self.file_lengths = self._get_file_lengths()

    def _get_file_lengths(self):
        """Get the length (number of samples) in each .npz file based on the length of one field (assume all fields are same length)."""
        lengths = []
        for file_path in self.file_paths:
            with np.load(file_path) as data:  # Open the file and extract the length of one field (e.g., 'action')
                lengths.append(len(data['action']))
        return lengths


    def __len__(self):
        """Returns the total number of samples across all files."""
        return sum(self.file_lengths)


    def _get_file_and_local_idx(self, idx):
        """Determine which file and index within that file corresponds to a global index."""
        running_total = 0
        for file_idx, length in enumerate(self.file_lengths):
            if running_total + length > idx:
                local_idx = idx - running_total
                return file_idx, local_idx
            running_total += length

    def __getitem__(self, idx):
        """Gets a single sample from the dataset at the given global index."""
        file_idx, local_idx = self._get_file_and_local_idx(idx)

        # Load the specific file
        with np.load(self.file_paths[file_idx], mmap_mode='r') as data:
            # Extract the relevant fields for the sample
            vis_obs = torch.tensor(data['vis_obs'][local_idx], dtype=torch.float)
            tac_obs = torch.tensor(data['tac_obs'][local_idx], dtype=torch.float)
            actions = torch.tensor(data['action'][local_idx], dtype=torch.float)
            gt_labels = torch.tensor(data['gt_obs'][local_idx], dtype=torch.float)

        # Return all the fields as a tuple
        return vis_obs, tac_obs, actions, gt_labels


class CrossModal(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
           file_paths (list of str): List of file paths to load the data.
        """
        self.file_paths = file_paths
        self.all_vis_obs, self.all_tac_obs, self.all_actions, self.all_gt_obs = self._load_data_into_ram()

    def _load_data_into_ram(self):
        """Load all npz files into RAM at initialization."""
        all_actions = []
        all_vis_obs = []
        all_tac_obs = []
        all_gt_obs = []

        for file_path in self.file_paths:
            with np.load(file_path) as data:
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