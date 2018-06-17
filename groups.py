"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# STD
import random
import math

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

# PROJECT
from activations import ActivationsDataset
from visualization import plot_multiple_model_weights


def _split(length: int, ratio=(0.9, 0.05, 0.05), seed=1):

    if not sum(ratio) == 1: raise ValueError('Ratios must sum to 1!')

    train_cutoff = math.floor(length*ratio[0])
    valid_cutoff = math.floor(length*(ratio[0]+ratio[1]))

    indices = list(range(length))
    random.seed(seed)
    random.shuffle(indices)
    train_indices = indices[:train_cutoff]
    valid_indices = indices[train_cutoff:valid_cutoff]
    test_indices = indices[valid_cutoff:]

    return train_indices, valid_indices, test_indices


class FunctionalGroupsDataset(ActivationsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_feature_label_added = False
        self.presence_column = []

    @staticmethod
    def load(load_path, convert_to_numpy=False):
        dataset = ActivationsDataset.load(load_path, convert_to_numpy)

        return FunctionalGroupsDataset(
            dataset.model_inputs, dataset.model_outputs,
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

            for model_input in self.model_inputs:
                occurrences = model_input == target_feature

                # Check whether the target feature appears in the input sequence and whether it is at a specific
                # position if position_sensitive > -1
                if occurrences.any() and (position_sensitive == -1 or occurrences[:, position_sensitive]):
                    class_label = torch.Tensor([1])
                else:
                    class_label = torch.Tensor([0])

                self.presence_column.append(class_label)

            # Overwrite data using the new class label column
            self.data = list(zip(getattr(self, target_activations), self.presence_column))

            self.target_feature_label_added = True


class DiagnosticBinaryClassifier(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input_):
        out = self.linear(input_)
        out = F.sigmoid(out)

        return out


def train(model: DiagnosticBinaryClassifier, train_data_loader, valid_data_loader, test_data_loader,
          criterion, optimizer, epochs=10):

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


if __name__ == "__main__":
    # Load data and split into sets
    seed = 12345
    num_models = 10
    full_dataset = FunctionalGroupsDataset.load("./ga_gru_1_heldout_tables.pt")
    full_dataset.add_target_feature_label(
        target_feature=18, target_activations="hidden_activations_decoder"
    )

    training_indices, validation_indices, test_indices = _split(len(full_dataset), ratio=(0.8, 0.1, 0.1))

    training_data_loader = DataLoader(
        dataset=full_dataset,
        sampler=SubsetRandomSampler(training_indices)
    )
    validation_data_loader = DataLoader(
        dataset=full_dataset,
        sampler=SubsetRandomSampler(validation_indices)
    )
    test_data_loader = DataLoader(
        dataset=full_dataset,
        sampler=SubsetRandomSampler(test_indices),
        batch_size=1
    )
    torch.manual_seed(12345)

    # Prepare model for training
    models = []
    for i in range(num_models):
        print("\nTraining model {}...\n".format(i+1))
        input_size = full_dataset.hidden_activations_decoder[0][0].size()[2]
        model = DiagnosticBinaryClassifier(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)

        # Train
        train(model, training_data_loader, validation_data_loader, test_data_loader, criterion, optimizer, epochs=50)
        models.append(model)

    # Plotting
    plot_multiple_model_weights(weights_to_plot=[model.linear.weight.detach().numpy() for model in models])

