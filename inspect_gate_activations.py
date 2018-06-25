from collections import defaultdict
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import os

from activations import ActivationsDataset

dataset = ActivationsDataset.load('data/gate_datasets/baseline_gru_run_1_sample_1_heldout_compositions_longer_compositions.pt')
colours = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

gru_labels = [
    'input_gate_decoder',
    'input_gate_encoder',
    'new_gate_decoder',
    'new_gate_encoder',
    'reset_gate_decoder',
    'reset_gate_encoder',
]

lstm_labels = [
    'input_gate_encoder',
    'forget_gate_encoder',
    'output_gate_encoder',
    'cell_gate_encoder',
    'forget_gate_decoder',
    'cell_gate_decoder',
    'input_gate_decoder',
    'output_gate_decoder',
]

for dataset_path in os.listdir('data/gate_datasets/'):

    dataset = ActivationsDataset.load('data/gate_datasets/' + dataset_path)

    if 'lstm' in dataset_path:
        gate_labels = lstm_labels
    else:
        gate_labels = gru_labels

    use = [index for index, label in enumerate(gate_labels) if 'encoder' in label]

    # left_saturated = [0] * 512
    # right_saturated = [0] * 512

    left_saturated_gates = defaultdict(lambda: [0] * 512)
    right_saturated_gates = defaultdict(lambda: [0] * 512)

    for datapoint in dataset:

        for index, gate in enumerate(datapoint[2:]):

            if not index in use: continue

            # threshold values as in http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf
            left_saturated_gates[index] = map(add, left_saturated_gates[index], (datapoint[index][-1] < 0.1).numpy() / len(dataset))
            right_saturated_gates[index] = map(add, right_saturated_gates[index], (datapoint[index][-1] > 0.9).numpy() / len(dataset))

        # left_saturated = [count / len(dataset) for count in left_saturated]
        # right_saturated = [count / len(dataset) for count in right_saturated]

    for index, (left_saturated, right_saturated) in enumerate(zip(left_saturated_gates.values(), right_saturated_gates.values())):
        plt.scatter(list(left_saturated), list(right_saturated), alpha=0.2, color=colours[index], label=gate_labels[use[index]])

    plt.plot(list(np.arange(0, 1, 0.0001)), list(reversed(np.arange(0, 1, 0.0001))), color='black')
    plt.legend(loc='upper right')
    plt.title = dataset_path
    plt.xlabel('left saturated')
    plt.ylabel('right saturated')
    plt.show()
    plt.clf()