"""

"""

# EXT
from torch.utils.data import Dataset
from torch import Tensor
import torch
import numpy as np


class ActivationsDataset(Dataset):
    """
    Simple dataset class to store all kinds of activations created by a certain input.
    """
    def __init__(self, model_inputs: list, model_outputs: list, **activations):
        """
        Initialize the data set.
        """
        assert all([type(input_seq) in (Tensor, np.ndarray) for input_seq in model_inputs]), "input_seqs has to be list of pytorch tensors"
        assert all([type(timestep) in (Tensor, np.ndarray) for activation_list in activations.values() for sample in activation_list for timestep in sample]), \
            "Activations are a list of lists of pytorch tensors (List of activations per time step per sample)."

        self.model_inputs = model_inputs
        self.model_outputs = model_outputs

        self.activation_columns = list(activations.keys())
        self.columns = ["model_inputs", "model_outputs"] + self.activation_columns

        for name, activation_list in activations.items():
            setattr(self, name, activation_list)

        self.data = list(zip(model_inputs, *activations.values()))
        self.length = len(model_inputs)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item]

    def save(self, save_path):
        torch.save(self, save_path)

    @staticmethod
    def load(load_path, convert_to_numpy=False):
        dataset = torch.load(load_path)

        def _squeeze_out(array):
            while len(array.shape) > 1:
                array = array.squeeze(0)
            return array

        if convert_to_numpy:
            for activation_column in dataset.columns:
                activations = getattr(dataset, activation_column)
                converted_activations = [
                    np.array([_squeeze_out(time_step.cpu().numpy()) for time_step in sample])
                    for sample in activations
                ]
                setattr(dataset, activation_column, converted_activations)

        return dataset
