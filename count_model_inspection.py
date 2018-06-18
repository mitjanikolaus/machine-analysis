import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from count_prediction import Regression

model = Regression(512)
weights = []
biases = []
model_names = []

model_directory = 'results/models/'

for file in os.listdir(model_directory):
    model.load_state_dict(torch.load(model_directory + file))

    weights.append(deepcopy(model.linear._parameters['weight'].data.abs()))
    biases.append(deepcopy(model.linear._parameters['bias'].data.abs()))

    model_names.append(file.split('.')[0])

minimum_value = min(float(min([torch.min(w) for w in weights])), float(min([torch.min(b) for b in biases])))
maximum_value = max(float(max([torch.min(w) for w in weights])), float(max([torch.min(b) for b in biases])))

for weight, bias, model_name in zip(weights, biases, model_names):
    plt.imshow(
        np.reshape(torch.cat((torch.unsqueeze(bias, dim=0), weight), 1).numpy(), (19, 27)),
        cmap='viridis', interpolation='nearest',
        # vmax=maximum_value, vmin=minimum_value
    )

    plt.savefig('results/weight_plots/{}.png'.format(model_name))
    plt.clf()

