"""
Define different utility functions, e.g. for different kinds of visualization.
"""

# EXT
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

# PROJECT
from activations import ActivationsDataset


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


def plot_activation_distributions(all_timesteps_activations: np.array, grid_size=None, show_title=True):
    num_timesteps = len(all_timesteps_activations)

    assert all([type(out) in (torch.Tensor, np.ndarray) for out in all_timesteps_activations]), \
        "This function only takes all the activations for all the time steps of a single sample."
    if grid_size:
        assert grid_size[0] * grid_size[1] >= num_timesteps, \
            "Specified grid doesn't provide enough space for all plots."

    def _hex_to_RGB(hex):
        # Pass 16 to the integer function for change of base
        return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]

    def _color_gradient(start_hex, finish_hex="#FFFFFF", n=10):
        """ Taken from https://bsou.io/posts/color-gradients-with-python and slightly modified. """
        # Starting and ending colors in RGB form
        s = _hex_to_RGB(start_hex)
        f = _hex_to_RGB(finish_hex)
        # Initilize a list of the output colors with the starting color
        RGB_list = [np.array(s) / 255]
        # Calcuate a color at each evenly spaced value of t from 1 to n
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = np.array([
                int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j]))
                for j in range(3)
            ]) / 255
            # Add it to our list of output colors
            RGB_list.append(curr_vector)

        return RGB_list

    if grid_size is None:
        grid_size = (1, num_timesteps)

    fig, axes = plt.subplots(*grid_size, sharey=True)
    colors = _color_gradient("#a8bee3", "#e58625", n=num_timesteps)

    for t, (axis, activations, color) in enumerate(zip(axes, all_timesteps_activations, colors)):
        num_bins = int(len(activations) / 4)
        axis.hist(activations, num_bins, density=True, facecolor=color, alpha=0.75)
        axis.set_xlabel("t={}".format(t))

    fig.tight_layout()

    if show_title:
        fig.suptitle("Distribution of activation values over {} time steps".format(num_timesteps))

    plt.subplots_adjust(top=0.85)
    plt.show()


def plot_activation_gradients(all_timesteps_activations: np.array, neuron_heatmap_size: tuple, show_title=True, absolute=True):
    num_timesteps = len(all_timesteps_activations)

    assert all([type(out) in (torch.Tensor, np.ndarray) for out in all_timesteps_activations]), \
        "This function only takes all the activations for all the time steps of a single sample."

    fig = plt.figure()
    last_activations = all_timesteps_activations[0]

    grid = AxesGrid(
        fig, 111, nrows_ncols=(1, num_timesteps-1), axes_pad=0.05, share_all=True, label_mode="L",
        cbar_location="right", cbar_mode="single",
    )

    for t, (axis, current_activations) in enumerate(zip(grid, all_timesteps_activations[1:])):
        activation_gradients = current_activations - last_activations
        vmin, vmax = -2, 2
        colormap = 'coolwarm'

        if absolute:
            vmin = 0
            colormap = "Reds"

        heatmap = axis.imshow(activation_gradients.reshape(*neuron_heatmap_size), cmap=colormap, vmin=vmin, vmax=vmax)
        axis.set_xlabel("t={} -> t={}".format(t, t+1))
        axis.set_xticks([])
        axis.set_yticks([])

        last_activations = current_activations

    grid.cbar_axes[0].colorbar(heatmap)

    if show_title:
        fig.suptitle("Activation value gradients over {} time steps".format(num_timesteps))

    plt.show()


if __name__ == "__main__":
    test_data_path = './ga_gru_1_heldout_tables.pt'
    data = ActivationsDataset.load(test_data_path, convert_to_numpy=True)
    target_activations = "hidden_activations_decoder"

    sample = getattr(data, target_activations)[12]  # Specific sample
    average = np.array(getattr(data, target_activations)).mean(axis=0)

    # Plot activations as heat map
    #encoder_input_length = data.model_inputs[0].shape[1]-1
    #plot_hidden_activations(data.encoder_activations[0], encoder_input_length, num_units_to_plot=50)

    # Plot distribution of activation values in a series of time steps
    #plot_activation_distributions(SAMPLE)

    # Plot changes in activations values
    #plot_activation_gradients(sample, neuron_heatmap_size=(16, 32))
    plot_activation_gradients(average, neuron_heatmap_size=(16, 32), show_title=False, absolute=True)
