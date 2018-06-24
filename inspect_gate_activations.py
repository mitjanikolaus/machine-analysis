from collections import defaultdict, Counter
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import os
from activations import ActivationsDataset

# type = 'gru'
type = 'lstm'

part = 'encoder'
# part = 'decoder'

# model = 'guided'
model = 'baseline'

colours = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

def gate2colour(gate_name: str) -> str:
    if 'input_gate' in gate_name: return 'green'
    # if 'new_gate' in gate_name: return 'blue'
    if 'reset_gate' in gate_name: return 'red'
    if 'forget' in gate_name: return 'red'
    if 'output' in gate_name: return 'blue'
    # if 'cell' in gate_name: return 'yellow'
    raise ValueError('Unknown gate name!')

gru_labels = [
    'input_gate_decoder',
    'input_gate_encoder',
    '--',
    '--',
    'reset_gate_decoder',
    'reset_gate_encoder',
]

lstm_labels = [
    'input_gate_encoder',
    'forget_gate_encoder',
    'output_gate_encoder',
    '--',
    'forget_gate_decoder',
    '--',
    'input_gate_decoder',
    'output_gate_decoder',
]


left_saturated_gates = defaultdict(lambda: [0] * 512)
right_saturated_gates = defaultdict(lambda: [0] * 512)
sanity_check = [0] * 512
counts = Counter()

for dataset_path in os.listdir('data/gate_datasets/'):

    if not type in dataset_path: continue
    if not model in dataset_path: continue

    dataset = ActivationsDataset.load('data/gate_datasets/' + dataset_path)
    print(dataset_path)

    if 'lstm' in dataset_path:
        gate_labels = lstm_labels
    else:
        gate_labels = gru_labels

    use = [index for index, label in enumerate(gate_labels) if part in label]

    for datapoint in dataset:

        for gate_index, gates_at_timesteps in enumerate(datapoint[1:]):

            if not gate_index in use: continue

            for gates_at_timestep in gates_at_timesteps:
                # threshold values as in http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf
                left_saturated_gates[gate_index] = list(map(add, left_saturated_gates[gate_index], (gates_at_timestep.view(-1) < 0.1).numpy()))
                right_saturated_gates[gate_index] = list(map(add, right_saturated_gates[gate_index], (gates_at_timestep.view(-1) > 0.9).numpy()))
                counts[gate_index] += 1


def display_label(gate_label):
    if 'input' in gate_label: return 'Input gate'
    if 'output' in gate_label: return 'Output gate'
    if 'forget' in gate_label: return 'Forget gate'
    if 'reset' in gate_label: return 'Reset gate'
    raise ValueError('unknown gate label')

for gate_index, (left_saturated, right_saturated) in enumerate(zip(left_saturated_gates.values(), right_saturated_gates.values())):
    plt.scatter(
        [gate / counts[use[gate_index]] for gate in list(left_saturated)],
        [gate / counts[use[gate_index]] for gate in list(right_saturated)],
        alpha=0.2, color=gate2colour(gate_labels[use[gate_index]]), label=display_label(gate_labels[use[gate_index]])
    )

plt.plot(list(np.arange(0, 1, 0.0001)), list(reversed(np.arange(0, 1, 0.0001))), color='black')
plt.legend(loc='upper right')
plt.xlabel('left saturated')
plt.ylabel('right saturated')
plt.savefig('../report/figures/gate_saturations_{}_{}_{}.pdf'.format(part, model, type))
plt.clf()