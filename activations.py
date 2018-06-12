"""
Module defining a data set class to store hidden activations.
"""

# EXT
from torch.utils.data import Dataset


class ActivationsDataset(Dataset):

    def __init__(self, input_seqs: list, **activations):
        self.input_seqs = input_seqs

        for name, activation_list in activations.items():
            setattr(self, name, activation_list)



