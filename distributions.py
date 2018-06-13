"""
Module implementing functions to understand how activation values changes for different models
over multiple time steps.
"""

# EXT
import numpy as np

# PROJECT
from activations import ActivationsDataset

if __name__ == "__main__":
    test_data_path = './baseline_gru_1_heldout_tables.pt'
    data = ActivationsDataset.load(test_data_path, convert_to_numpy=True)
    target_activations = "hidden_activations_encoder"

    sample = getattr(data, target_activations)[12]  # Specific sample
    average = np.array(getattr(data, target_activations)).mean(axis=0)
