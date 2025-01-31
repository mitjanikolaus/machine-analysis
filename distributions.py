"""
Module implementing functions to understand how activation values changes for different models
over multiple time steps.
"""

# STD
from collections import defaultdict
from random import sample

# EXT
import numpy as np
from numpy.linalg import norm
from scipy.stats import kurtosis, skew, ttest_ind


# PROJECT
from activations import ActivationsDataset


def analyze_activation_distributions(all_timesteps_activations: np.array, moments=["mean", "variance"]):
    all_timesteps_distributions = defaultdict(list)
    moment_functions = {"mean": np.mean, "variance": np.var, "skewness": skew, "kurtosis": kurtosis}

    for current_activations in all_timesteps_activations:
        for moment in moments:
            all_timesteps_distributions[moment].append(moment_functions[moment](current_activations))

    return dict(all_timesteps_distributions)


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
    baseline_lstm_data_path = './data/baseline_lstm_1_heldout_tables.pt'
    baseline_gru_data_path = './data/baseline_gru_1_heldout_tables.pt'
    ga_lstm_data_path = './data/ga_lstm_1_heldout_tables.pt'
    ga_gru_data_path = './data/ga_gru_1_heldout_tables.pt'
    data_set_paths = [baseline_lstm_data_path, baseline_gru_data_path, ga_lstm_data_path, ga_gru_data_path]

    # Load everything
    baseline_lstm_data = ActivationsDataset.load(baseline_lstm_data_path, convert_to_numpy=True)
    baseline_gru_data = ActivationsDataset.load(baseline_gru_data_path, convert_to_numpy=True)
    ga_lstm_data = ActivationsDataset.load(ga_lstm_data_path, convert_to_numpy=True)
    ga_gru_data = ActivationsDataset.load(ga_gru_data_path, convert_to_numpy=True)
    data_sets = [baseline_lstm_data, baseline_gru_data, ga_lstm_data, ga_gru_data]

    # Prepare for experiments
    num_samples = 3
    target_activations = "hidden_activations_decoder"
    #sample_indices = sample(range(len(baseline_gru_data)), num_samples)
    sample_indices = [39, 52, 87]

    # Activation distributions experiments
    # print("\n### Activation distributions ###\n")
    # for path, data_set in zip(data_set_paths, data_sets):
    #     average_activations = np.array(getattr(data_set, target_activations)).mean(axis=0)
    #     print("Results for {}:".format(path))
    #     print("Distributions for the average of all samples", analyze_activation_distributions(average_activations))
    #
    #     for sample_index in sample_indices:
    #         print(
    #             "Norm for sample #{}:".format(sample_index),
    #             analyze_activation_distributions(getattr(data_set, target_activations)[sample_index])
    #         )
    #     print("")
    #
    # # Activation gradient experiments
    # print("\n### Activation gradients ###\n")
    # for path, data_set in zip(data_set_paths, data_sets):
    #     average_activations = np.array(getattr(data_set, target_activations)).mean(axis=0)
    #     print("Results for {}:".format(path))
    #     print("Norm the avg of samples", analyze_activation_gradients(average_activations))
    #
    #     for sample_index in sample_indices:
    #         print(
    #             "Norm for sample #{}:".format(sample_index),
    #             analyze_activation_gradients(getattr(data_set, target_activations)[sample_index])
    #         )
    #     print("")

    # Check if results are significant for all samples per time step
    all_magnitudes = []

    for data_set in data_sets:
        samples = getattr(data_set, target_activations)
        magnitudes = np.empty((0, samples[0].shape[0] - 1))  # Number of time steps minus one

        for current_sample in samples:
            current_magnitudes = analyze_activation_gradients(current_sample)
            current_magnitudes = np.array(current_magnitudes).reshape(1, -1)
            magnitudes = np.concatenate((magnitudes, current_magnitudes), axis=0)

        all_magnitudes.append(magnitudes)

    for bl_data, ga_data in [(all_magnitudes[0], all_magnitudes[2]), (all_magnitudes[1], all_magnitudes[3])]:
        num_transitions = bl_data.shape[1]  # Number of transitions between two time steps

        for transition in range(num_transitions):
            _, p_value = ttest_ind(bl_data[:, transition], ga_data[:, transition])
            print("p value between time steps {} and {} is {}".format(transition+1, transition+2 , p_value))
