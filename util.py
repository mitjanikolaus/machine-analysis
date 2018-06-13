"""
Define different utility functions, e.g. for different kinds of visualization.
"""

# EXT
import torch
import matplotlib.pyplot as plt
import numpy as np

# PROJECT
from activations import ActivationsDataset
from models.analysable_decoder import HiddenStateAnalysisDecoderRNN


def plot_hidden_activations(activations, input_length, num_units_to_plot=50):
    """
    Plot hidden activations for 1 sample
    """
    for ts in range(len(activations)):
        activations[ts] = activations[ts].numpy()

    activations = np.array(activations).reshape(-1, input_length)

    activations = activations[np.random.choice(activations.shape[0], num_units_to_plot),:]

    heatmap = plt.imshow(activations, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(0, input_length, step=1))
    plt.xlabel('timestep')
    plt.ylabel('hidden unit')
    plt.colorbar(heatmap)
    plt.show()


def plot_activation_distributions(all_timesteps_activations: list, grid_size=None):
    assert all([type(out) == torch.Tensor for out in all_timesteps_activations]), \
        "This function only takes all the activations for all the time steps of a single sample."
    if grid_size:
        assert grid_size[0] * grid_size[1] >= len(all_timesteps_activations), \
            "Specified grid doesn't provide enough space for all plots."

    def _squeeze_out(array):
        while len(array.shape) > 1:
            array = array.squeeze(0)
        return array

    if grid_size is None:
        grid_size = (1, len(all_timesteps_activations))

    fig, axes = plt.subplots(*grid_size, sharey=True)

    for axis, activations in zip(axes, all_timesteps_activations):
        activations = activations.cpu().numpy()
        activations = _squeeze_out(activations)
        print(activations)
        num_bins = int(len(activations) / 4)
        n, bins, patches = axis.hist(activations, num_bins, density=True, facecolor='g', alpha=0.75)

    fig.tight_layout()

    plt.show()







if __name__ == "__main__":
    test_data_path = './test_activations_lstm_1_heldout_tables.pt'
    data = ActivationsDataset.load(test_data_path)

    # Plot activations as heat map
    #encoder_input_length = data.model_inputs[0].shape[1]-1
    #plot_hidden_activations(data.encoder_activations[0], encoder_input_length, num_units_to_plot=50)

    # Plot distribution of activation values in a series of time steps
    plot_activation_distributions(data.hidden_activations_encoder[1])
