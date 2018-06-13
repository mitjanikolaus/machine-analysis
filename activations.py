"""

"""

# EXT
from torch.utils.data import Dataset
from torch import Tensor
import torch


class ActivationsDataset(Dataset):
    """
    Simple dataset class to store all kinds of activations created by a certain input.
    """
    def __init__(self, model_inputs: list, model_outputs: list, **activations):
        """
        Initialize the data set.
        """
        assert all([type(input_seq) == Tensor for input_seq in model_inputs]), "input_seqs has to be list of pytorch tensors"
        assert all([type(timestep) == Tensor for activation_list in activations.values() for sample in activation_list for timestep in sample]), \
            "Activations are a list of lists of pytorch tensors (List of activations per time step per sample)."

        self.model_inputs = model_inputs
        self.model_outputs = model_outputs

        self.columns = ["Inputs", "Outputs"] + list(activations.keys())  # Names of columns in data set

        for name, activation_list in activations.items():
            setattr(self, name, activation_list)

        self.data = zip(model_inputs, *activations.values())
        self.length = len(model_inputs)

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
