"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# STD
import random
import math
from collections import Counter

# EXT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import numpy as np

# PROJECT
from activations import ActivationsDataset
from visualization import plot_multiple_model_weights


def _split(length: int, ratio=(0.9, 0.05, 0.05)):

    if not sum(ratio) == 1: raise ValueError('Ratios must sum to 1!')

    train_cutoff = math.floor(length*ratio[0])
    valid_cutoff = math.floor(length*(ratio[0]+ratio[1]))

    indices = list(range(length))
    random.shuffle(indices)
    train_indices = indices[:train_cutoff]
    valid_indices = indices[train_cutoff:valid_cutoff]
    test_indices = indices[valid_cutoff:]

    return train_indices, valid_indices, test_indices



class FunctionalGroupsDataset(ActivationsDataset):
    """
    Data set tailored to identify functional groups of neurons.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_feature_label_added = False
        self.presence_column = []
        self.selected_decoder_hidden_states = []

    @staticmethod
    def load(load_path, convert_to_numpy=False):
        dataset = ActivationsDataset.load(load_path, convert_to_numpy)

        return FunctionalGroupsDataset(
            dataset.model_inputs, dataset.model_outputs,
            # Awkward line: Initialize the same data set again, this time only as a FunctionalGroupDataset object
            **dict(zip(
                dataset.activation_columns,
                [getattr(dataset, activation_column) for activation_column in dataset.activation_columns]
            ))
        )

    def add_target_feature_label(self, target_feature: int, target_activations: str, position_sensitive=-1):
        """
        Add a binary label to every instance in the data set, telling whether a target feature is present in the input
        sequence (and if it is at a specified position given that position_sensitive > -1).
        """
        if not self.target_feature_label_added:
            # Column describing whether the target feature is present in input sequence
            self.columns = [target_activations, "target_feature_present"]

            for model_input, decoder_hidden_states in zip(self.model_inputs, getattr(self, target_activations)):
                occurrences = model_input == target_feature
                occurrence_index = occurrences.squeeze().nonzero()  # Index of timestep where input occurred

                # Target feature not in input: Select random decoder output
                if len(occurrence_index) == 0:
                    occurrence_index = random.sample(range(len(decoder_hidden_states)), k=1)[0]
                # Target feature in input: Take decoder hidden time step corresponding to input time step
                else:
                    occurrence_index = occurrence_index[0]

                # Check whether the target feature appears in the input sequence and whether it is at a specific
                # position if position_sensitive > -1
                if occurrences.any() and (position_sensitive == -1 or occurrences[:, position_sensitive]):
                    class_label = torch.Tensor([1])
                else:
                    class_label = torch.Tensor([0])

                self.presence_column.append(class_label)
                self.selected_decoder_hidden_states.append(decoder_hidden_states[occurrence_index])

            # Overwrite data using the new class label column
            self.data = list(zip(self.selected_decoder_hidden_states, self.presence_column))

            self.target_feature_label_added = True  # Only allow this logic to be called once

    def add_target_feature_label_regressor(self, target_feature: int, target_activations: str, position_sensitive=-1):
        """
        Add a binary label to every instance in the data set, telling whether a target feature is present in the input
        sequence (and if it is at a specified position given that position_sensitive > -1).
        """
        if not self.target_feature_label_added:
            # Column describing whether the target feature is present in input sequence
            self.columns = [target_activations, "target_feature_present"]

            for model_input, decoder_hidden_states in zip(self.model_inputs, getattr(self, target_activations)):
                occurrences = model_input == target_feature
                occurrence_index = occurrences.squeeze().nonzero()[0]  # Index of timestep where input occurred

                # TODO include all input samples!
                # Target feature not in input: Select random decoder output
                if len(occurrence_index) == 0:
                    occurrence_index = random.sample(range(len(decoder_hidden_states)), k=1)[0]
                # Target feature in input: Take decoder hidden time step corresponding to input time step
                else:
                    occurrence_index = occurrence_index[0]

                # Check whether the target feature appears in the input sequence and whether it is at a specific
                # position if position_sensitive > -1
                if occurrences.any() and (position_sensitive == -1 or occurrences[:, position_sensitive]):
                    class_label = torch.Tensor([1])
                else:
                    class_label = torch.Tensor([0])

                self.presence_column.append(class_label)
                self.selected_decoder_hidden_states.append(decoder_hidden_states[occurrence_index].reshape(-1))

            # Overwrite data using the new class label column
            self.selected_decoder_hidden_states = np.array(self.selected_decoder_hidden_states)
            self.presence_column = np.array(self.presence_column)

            self.target_feature_label_added = True  # Only allow this logic to be called once


class DiagnosticBinaryClassifier(nn.Module):
    """
    Very basic binary classifier that tries to predict the occurrence of a specific input token based on the
    decoder hidden state, corresponding to the time step of the input token.
    """
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input_):
        out = self.linear(input_)
        out = F.sigmoid(out)  # 0 -> Token not in input, 1 -> Token in input

        return out


def train(model: DiagnosticBinaryClassifier, train_data_loader, valid_data_loader, test_data_loader, criterion,
          optimizer, epochs=10):

    for epoch in range(epochs):
        total_train_loss = 0
        total_valid_loss = 0
        test_accuracy = 0

        # Training set
        for activations, label in train_data_loader:
            activations = activations[-1]  # Take final time step activation

            optimizer.zero_grad()
            model.zero_grad()

            out = model.forward(activations)
            loss = criterion(out, label)

            total_train_loss += loss

            loss.backward()
            optimizer.step()

        # Validation set
        for activations, label in valid_data_loader:
            activations = activations[-1]  # Take final time step activation

            out = model.forward(activations)
            loss = criterion(out, label)

            total_valid_loss += loss

        # Test set
        for activations, label in test_data_loader:
            activations = activations[-1]  # Take final time step activation

            out = model.forward(activations)
            prediction = 1 if out.detach().numpy() > 0.5 else 0

            if prediction == label:
                test_accuracy += 1

        test_accuracy /= len(test_data_loader)
        test_accuracy *= 100

        print("\n### EPOCH {} ###".format(epoch + 1))
        print("Training loss for epoch #{}/{}: {:.2f}".format(epoch+1, epochs, total_train_loss))
        print("Validation loss for epoch #{}/{}: {:.2f}".format(epoch+1, epochs, total_valid_loss))
        print("Test accuracy for epoch #{}/{}: {:.2f} %".format(epoch+1, epochs, test_accuracy))


def print_correlation_matrix(models_weights):
    """
    Print a matrix of Pearson's rhos, showing all possible degrees of correlation between a pair of model weights.
    """
    rhos = []
    for (model_A, model_A_weights) in enumerate(models_weights):
        row = []

        for (model_B, model_B_weights) in enumerate(models_weights):
            rho, _ = pearsonr(model_A_weights.squeeze(), model_B_weights.squeeze())
            row.append("{:.2f}".format(rho))

        rhos.append(row)

    row_format = "{:>10}" * (len(models) + 1)
    print(row_format.format("", *range(len(models))))
    for model_num, row in zip(range(len(models)), rhos):
        print(row_format.format(model_num, *row))


def test_neuron_significance(models_weights, p=0.01):
    """
    Test whether some neurons responded in a statistically significant manner to an input based on the weights of
    several independently trained diagnostic classifiers. The test is a combination of a two-tailed and a one-tailed
    significance test:

    1. (For every single DC) Detect neurons that have been assigned significant weight values by checking whether for
    their weight w holds w > mu + 2 * sigma or w < mu - 2 * mu, effectively placing them in the 2.2 % of either side
    of the weight value distribution.

    2. (Across all DCs) Check how often the DCs agree on deeming neurons significant based on criterion 1. and see if
    the number of agreements is higher than for 95 % of all other neurons.
    """
    # 1. Step: Find candidate neurons
    candidate_neurons = [
        list(np.where(np.abs(model_weights) > model_weights.mean() + 2 * model_weights.std())[1])
        for model_weights in models_weights
    ]

    # 2. Step: Count agreements for each neurons
    num_neurons = models_weights[0].shape[1]
    agreements = Counter({neuron: 0 for neuron in range(num_neurons)})

    for classifier_candidates in candidate_neurons:
        agreements.update(classifier_candidates)

    # TODO: This doesn't work, selects too many neurons
    # Find the neurons that account for 1 - p per cent of all agreements. Those are the significant ones
    cumulative_agreements = 0
    total_agreements = sum(agreements.values())
    significant_neurons = []
    sorted_agreements = sorted(agreements.items(), key=lambda x: x[1], reverse=True)

    for neuron, neuron_agreements in sorted_agreements:
        significant_neurons.append((neuron, neuron_agreements))
        cumulative_agreements += neuron_agreements

        if cumulative_agreements > (1 - p) * total_agreements:
            break

    print(significant_neurons)
    return significant_neurons


if __name__ == "__main__":
    # Load data and split into sets
    num_models = 10
    epochs = 50
    target_feature = 3  # t1 = 3

    full_dataset = FunctionalGroupsDataset.load("./guided_gru_1_train.pt", convert_to_numpy=True)

    full_dataset.add_target_feature_label_regressor(
        target_feature=target_feature, target_activations="hidden_activations_decoder", position_sensitive=-1
    )

    for i in range(num_models):
        #create a new random split for each model
        training_indices, validation_indices, test_indices = _split(len(full_dataset), ratio=(0.9, 0.1, 0))

        X = full_dataset.selected_decoder_hidden_states
        y = full_dataset.presence_column

        regressor = LogisticRegression()
        regressor.fit(X[training_indices], y[training_indices])
        print(regressor.score(X[validation_indices], y[validation_indices]))

    """
    # Plotting
    models_weights = [model.linear.weight.detach().numpy() for model in models]
    # print_correlation_matrix(models_weights)
    # plot_multiple_model_weights(weights_to_plot=models_weights)
    test_neuron_significance(models_weights)
    """



