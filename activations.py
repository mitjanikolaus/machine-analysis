"""

"""

from copy import deepcopy

# EXT
from torch.utils.data import Dataset
from torch import Tensor
import torch


class ActivationsDataset(Dataset):
    """
    Simple dataset class to store all kinds of activations created by a certain input.
    """
    def __init__(self, input_seqs: list, **activations):
        """
        Initialize the data set.

        :param input_seqs: List of inputs that caused the activations.
        :param activations:
        """
        assert all([type(input_seq) == Tensor for input_seq in input_seqs]), "input_seqs has to be list of pytorch tensors"
        assert all([type(timestep) == Tensor for activation_list in activations.values() for sample in activation_list for timestep in sample]), \
            "Activations are a list of lists of pytorch tensors (List of activations per time step per sample)."

        self.input_seqs = input_seqs

        for name, activation_list in activations.items():
            setattr(self, name, activation_list)

        self.data = zip(input_seqs, *activations.values())
        self.length = len(input_seqs)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item]

    def save(self, save_path):
        torch.save(self, save_path)

    @staticmethod
    def load(load_path):
        dataset = torch.load(load_path)
        return dataset
