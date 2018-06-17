"""
Define different utility functions, e.g. for different kinds of visualization.
"""

# STD
import re

# EXT
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

# PROJECT
from activations import ActivationsDataset


def plot_hidden_activations(activations, input_length, num_units_to_plot=50):
    """
    Plot hidden activations for 1 sample
    """
    for ts in range(len(activations)):
        activations[ts] = activations[ts]

    activations = np.array(activations).reshape(-1, input_length)

    activations = activations[np.random.choice(activations.shape[0], num_units_to_plot),:]

    heatmap = plt.imshow(activations, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(0, input_length, step=1))
    plt.xlabel('timestep')
    plt.ylabel('hidden unit')
    plt.colorbar(heatmap)
    plt.show()


def plot_multiple_model_weights(weights_to_plot):
    """
    Plot the weights of different models side by side as a heat maps.
    """
    models_weights = np.array([rectangularfy(weights) for weights in weights_to_plot])
    vmax = models_weights.max()
    vmin = models_weights.min()

    fig = plt.figure()
    grid = AxesGrid(
        fig, 111, nrows_ncols=(1, len(models_weights)), axes_pad=0.05, share_all=True, label_mode="L",
        cbar_location="right", cbar_mode="single",
    )

    for axis, model_weights in zip(grid, models_weights):
        heatmap = axis.imshow(model_weights, cmap="coolwarm", vmax=vmax, vmin=vmin)
        axis.set_xticks([])
        axis.set_yticks([])

    grid.cbar_axes[0].colorbar(heatmap)

    plt.show()


def plot_activation_distributions(all_timesteps_activations: np.array, grid_size=None, show_title=True):
    """
    Plot the distribution of activation values over multiple time steps for a single sample.
    """
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

    fig, axes = plt.subplots(*grid_size, sharey=True, figsize=(5, 4))
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


def plot_activation_distributions_development(show_title=True, name_func=lambda name: name, **samples_all_time_step_activations):
    """
    Plot how the distributions of activation values change for multiple samples as a box and whiskers plot.
    """
    num_samples = len(samples_all_time_step_activations.keys())
    num_timesteps = len(list(samples_all_time_step_activations.values())[0])

    fig, axes = plt.subplots(nrows=1, ncols=num_timesteps, sharey=True)
    colors = cm.viridis(np.linspace(0, 1, num_samples))
    bplots = []

    for t, axis in enumerate(axes):
        bplot = axis.boxplot([
            all_time_step_activations[t] for all_time_step_activations in samples_all_time_step_activations.values()
        ], vert=True, sym="", patch_artist=True, whis=10000)  # Show min and max by setting whis very high
        axis.set_xlabel("t={}".format(t))
        axis.set_xticks([])
        bplots.append(bplot)

        # Coloring
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

    # Add legend
    fig.subplots_adjust(bottom=0.2)
    axes[int(num_timesteps/2)].legend(
        [bplot for bplot in bplots[0]["boxes"]],
        [name_func(name) for name in list(samples_all_time_step_activations.keys())],
        loc="lower left", bbox_to_anchor=(-2.75, -0.2), borderaxespad=0.1, ncol=num_samples
    )

    if show_title:
        fig.suptitle(
            "Distributions of activation values of {} samples over {} time steps".format(num_samples, num_timesteps)
        )

    plt.show()


def plot_activation_gradients(all_timesteps_activations: np.array, neuron_heatmap_size: tuple, show_title=True,
                              absolute=True, save=None):
    """
    Plot the changes in activation values between time steps as heat maps for one single sample.
    """
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

    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches="tight")


def get_name_from_activation_data_path(path):
    pattern = re.compile(".*\/((baseline|ga)_(lstm|gru))_.*")
    match = pattern.match(path).groups()[0]
    return match


def rectangularfy(array):
    """
    Try to reshape an numpy error so it gets as rectangular as possible.
    """
    assert 1 == array.shape[0]
    assert len(array.shape) == 2

    last_dim_mismatch = np.inf

    while True:
        try:
            new_array = array.reshape(int(array.shape[0] * 2), int(array.shape[1] / 2))
            current_mismatch = np.abs(array.shape[0] - array.shape[1])

            if current_mismatch >= last_dim_mismatch:
                break

            array = new_array
            last_dim_mismatch = current_mismatch
        except Exception as e:
            break

    return array


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

    # Prepare for experiments
    num_samples = 3
    target_activations = "forget_gate_activations_encoder"
    #sample_indices = sample(range(len(baseline_gru_data)), num_samples)
    sample_indices = [39, 52, 87]

    # Plot activations as heat map
    #encoder_input_length = baseline_lstm_data.model_inputs[0].shape[1]-1
    #plot_activation_distributions(baseline_lstm_data.hidden_activations_encoder[39], show_title=False)

    # Plot distribution of activation values in a series of time steps
    activation_dists_to_plot = {}
    for path, data_set in zip(data_set_paths, data_sets):
        average_activations = getattr(data_set, target_activations)[52]
        activation_dists_to_plot[path] = average_activations

    plot_activation_distributions_development(
        name_func=get_name_from_activation_data_path, show_title=False, **activation_dists_to_plot
    )


    # Plot changes in activations values
    #plot_activation_gradients(sample, neuron_heatmap_size=(16, 32))
    #plot_activation_gradients(
    #    getattr(ga_lstm_data, target_activations).mean(axis=0), neuron_heatmap_size=(16, 32), show_title=False, absolute=True,
    #)

    # for model_name, data_set in zip(["baseline_lstm", "baseline_gru", "ga_lstm", "ga_gru"], data_sets):
    #     for network_name, network_data in zip(["encoder", "decoder"], ["hidden_activations_encoder", "hidden_activations_decoder"]):
    #         for act_name, target_activations_func in zip(["39", "avg"], [lambda x: x[39], lambda x: x.mean(axis=0)]):
    #             plot_activation_gradients(
    #                 target_activations_func(getattr(data_set, network_data)), neuron_heatmap_size=(16, 32),
    #                 show_title=False, absolute=True,
    #                 save="./fig/gradients_{}_{}_{}.png".format(model_name, network_name, act_name)
    #             )
