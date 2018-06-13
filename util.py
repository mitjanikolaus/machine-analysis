import matplotlib.pyplot as plt
import numpy as np

from activations import ActivationsDataset
from models.HiddenStateAnalysisDecoderRNN import HiddenStateAnalysisDecoderRNN

#plot hidden activations for 1 sample
def plot_hidden_activations(activations, input_length, num_units_to_plot=50):
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


test_data_path = './test_activations.pt'
data = ActivationsDataset.load(test_data_path)
encoder_input_length = data.model_inputs[0].shape[1]-1
plot_hidden_activations(data.encoder_activations[0], encoder_input_length, num_units_to_plot=50)