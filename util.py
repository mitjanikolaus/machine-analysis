"""
Define different utility functions, e.g. for different kinds of visualization.
"""

# EXT
import torch
import torchtext

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

from seq2seq.dataset import SourceField, TargetField, AttentionField


# PROJECT
from activations import ActivationsDataset


def load_test_data(test_data_path, input_vocab, output_vocab, ignore_output_eos, use_attention_loss, attention_method, max_len):
    IGNORE_INDEX = -1
    output_eos_used = not ignore_output_eos

    src = SourceField()
    tgt = TargetField(output_eos_used)

    tabular_data_fields = [('src', src), ('tgt', tgt)]

    if use_attention_loss or attention_method == 'hard':
        attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
        tabular_data_fields.append(('attn', attn))

    src.vocab = input_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate test set
    test_set = torchtext.data.TabularDataset(
        path=test_data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    return test_set


def plot_activations(all_timesteps_activations: np.array, neuron_heatmap_size: tuple, show_title=True, absolute=True):
    num_timesteps = len(all_timesteps_activations)

    assert all([type(out) in (torch.Tensor, np.ndarray) for out in all_timesteps_activations]), \
        "This function only takes all the activations for all the time steps of a single sample."

    fig = plt.figure()

    grid = AxesGrid(
        fig, 111, nrows_ncols=(1, num_timesteps), axes_pad=0.05, share_all=True, label_mode="L",
        cbar_location="right", cbar_mode="single",
    )

    for t, (axis, current_activations) in enumerate(zip(grid, all_timesteps_activations[:])):
        vmin, vmax = -2, 2
        colormap = 'coolwarm'

        if absolute:
            vmin = 0
            colormap = "Reds"

        heatmap = axis.imshow(current_activations.reshape(*neuron_heatmap_size), cmap=colormap, vmin=vmin,
                              vmax=vmax)
        axis.set_xlabel("t={}".format(t))
        axis.set_xticks([])
        axis.set_yticks([])

    grid.cbar_axes[0].colorbar(heatmap)

    if show_title:
        fig.suptitle("Activation values over {} time steps".format(num_timesteps))

    plt.show()


def plot_activations_multiple_samples(all_timesteps_activations: np.array, neuron_heatmap_size: tuple, title, show_title=True, absolute=False):
    num_samples = len(all_timesteps_activations)
    num_timesteps = len(all_timesteps_activations[0])

    fig = plt.figure()

    grid = AxesGrid(
        fig, 111, nrows_ncols=(num_samples, num_timesteps), axes_pad=0.05, share_all=True, label_mode="L",
        cbar_location="right", cbar_mode="single",
    )

    for sample in range(num_samples):
        for t, (current_activations) in enumerate(all_timesteps_activations[sample][:]):
            axis = grid[(sample*num_timesteps)+t]
            vmin, vmax = -2, 2
            colormap = 'coolwarm'

            if absolute:
                vmin = 0
                colormap = "Reds"

            heatmap = axis.imshow(current_activations.reshape(*neuron_heatmap_size), cmap=colormap, vmin=vmin,
                                  vmax=vmax)
            axis.set_xlabel("t={}".format(t))
            axis.set_ylabel("sample {}".format(sample))
            axis.set_xticks([])
            axis.set_yticks([])

            grid.cbar_axes[0].colorbar(heatmap)

    if show_title:
        if title == None:
            fig.suptitle("Activation values over {} time steps".format(num_timesteps))
        else:
            fig.suptitle(title)

    plt.show()


def plot_activation_distributions(all_timesteps_activations: list, grid_size=None):
    num_timesteps = len(all_timesteps_activations)

    assert all([type(out) == torch.Tensor for out in all_timesteps_activations]), \
        "This function only takes all the activations for all the time steps of a single sample."
    if grid_size:
        assert grid_size[0] * grid_size[1] >= num_timesteps, \
            "Specified grid doesn't provide enough space for all plots."

    def _squeeze_out(array):
        while len(array.shape) > 1:
            array = array.squeeze(0)
        return array

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
        activations = activations.cpu().numpy()
        activations = _squeeze_out(activations)
        print(activations)
        num_bins = int(len(activations) / 4)
        axis.hist(activations, num_bins, density=True, facecolor=color, alpha=0.75)
        axis.set_xlabel("t={}".format(t))

    fig.tight_layout()
    fig.suptitle("Distribution of activation values over {} time steps".format(num_timesteps))

    plt.subplots_adjust(top=0.85)
    plt.show()



if __name__ == "__main__":
    test_data_path = './test_activations_lstm_1_heldout_tables.pt'
    data = ActivationsDataset.load(test_data_path)

    # Plot activations over time as heat maps
    plot_activations(data.hidden_activations_decoder[0], neuron_heatmap_size=(32, 16))

    # Plot distribution of activation values in a series of time steps
    #plot_activation_distributions(data.hidden_activations_decoder[12])

