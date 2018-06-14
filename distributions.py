"""
Module implementing functions to understand how activation values changes for different models
over multiple time steps.
"""

# STD
from random import sample

# EXT
import numpy as np
from numpy.linalg import norm

# PROJECT
from activations import ActivationsDataset


def analyze_activation_gradients(all_timesteps_activations: np.array,
                                 metric=lambda current, last: norm(np.abs(current - last))):
    last_activations = all_timesteps_activations[0]
    all_timesteps_gradients = []

    for t, current_activations in enumerate(all_timesteps_activations[1:]):
        activation_gradients = metric(current_activations, last_activations)
        all_timesteps_gradients.append(activation_gradients)

        last_activations = current_activations

    return all_timesteps_gradients


if __name__ == "__main__":
    # Path to activation data sets
    baseline_lstm_data_path = './baseline_lstm_1_heldout_tables.pt'
    baseline_gru_data_path = './baseline_gru_1_heldout_tables.pt'
    ga_lstm_data_path = './ga_lstm_1_heldout_tables.pt'
    ga_gru_data_path = './ga_gru_1_heldout_tables.pt'
    data_set_paths = [baseline_lstm_data_path, baseline_gru_data_path, ga_lstm_data_path, ga_gru_data_path]

    # Load everything
    baseline_lstm_data = ActivationsDataset.load(baseline_lstm_data_path, convert_to_numpy=True)
    baseline_gru_data = ActivationsDataset.load(baseline_gru_data_path, convert_to_numpy=True)
    ga_lstm_data = ActivationsDataset.load(ga_lstm_data_path, convert_to_numpy=True)
    ga_gru_data = ActivationsDataset.load(ga_gru_data_path, convert_to_numpy=True)
    data_sets = [baseline_lstm_data, baseline_gru_data, ga_lstm_data, ga_gru_data]

    # Do experiments
    num_samples = 3
    target_activations = "hidden_activations_encoder"
    sample_indices = sample(range(len(baseline_gru_data)), num_samples)

    for path, data_set in zip(data_set_paths, data_sets):
        average_activations = np.array(getattr(data_set, target_activations)).mean(axis=0)
        print("Results for {}:".format(path))
        print("Norm the avg of samples", analyze_activation_gradients(average_activations))

        for sample_index in sample_indices:
            print(
                "Norm for sample #{}:".format(sample_index),
                analyze_activation_gradients(getattr(data_set, target_activations)[sample_index])
            )
        print("")
