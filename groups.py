"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# STD
import random
import math
from collections import Counter

# EXT
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# PROJECT
from activations import ActivationsDataset


def _split(length: int, ratio=(0.9, 0.1)):

    if not sum(ratio) == 1: raise ValueError('Ratios must sum to 1!')

    train_cutoff = math.floor(length*ratio[0])

    indices = list(range(length))
    random.shuffle(indices)
    train_indices = indices[:train_cutoff]
    test_indices = indices[train_cutoff:]

    return train_indices, test_indices


class FunctionalGroupsDataset(ActivationsDataset):
    """
    Data set tailored to identify functional groups of neurons.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_feature_label_added = False
        self.regressor_presence_column = []
        self.regressor_decoder_hidden_states = []

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

    def add_dataset_for_regressor(self, target_feature: int, target_activations: str):
        """
        Add a binary label to every instance in the data set, telling whether a target feature is present in the input
        sequence (and if it is at a specified position given that position_sensitive > -1).
        """
        if not self.target_feature_label_added:
            # Column describing whether the target feature is present in input sequence
            self.columns = [target_activations, "target_feature_present"]
            num_positive = 0

            positive_decoder_hidden_states, negative_decoder_hidden_states = [], []

            for model_input, decoder_hidden_states in zip(self.model_inputs, getattr(self, target_activations)):
                occurrence_indices = np.where(model_input.flatten() == target_feature)[0]

                for ts, decoder_hidden_state in enumerate(decoder_hidden_states):
                    # Target feature in input: label 1
                    if ts in occurrence_indices:
                        num_positive += 1
                        positive_decoder_hidden_states.append(decoder_hidden_state)
                    else:
                        negative_decoder_hidden_states.append(decoder_hidden_state)

            # Balance data set
            regressor_decoder_hidden_states = positive_decoder_hidden_states + negative_decoder_hidden_states[
                                                                               :num_positive]
            regressor_class_labels = [1] * num_positive + [0] * num_positive

            # Overwrite data using the new class label column
            self.regressor_decoder_hidden_states = np.array(regressor_decoder_hidden_states)
            self.regressor_presence_column = np.array(regressor_class_labels)

            self.length = 2 * num_positive

            self.target_feature_label_added = True  # Only allow this logic to be called once

if __name__ == "__main__":
    target_feature = 3  # t1 = 3

    # Load data and split into sets
    full_dataset = FunctionalGroupsDataset.load("./guided_gru_1_train.pt", convert_to_numpy=True)
    full_dataset.add_dataset_for_regressor(
        target_feature=target_feature, target_activations="hidden_activations_decoder"
    )

    def train_regressor(X, y, training_indices, test_indices):
        regressor = LogisticRegression()
        regressor.fit(X[training_indices], y[training_indices])

        return regressor.score(X[test_indices], y[test_indices]), regressor.coef_


    X = full_dataset.regressor_decoder_hidden_states
    y = full_dataset.regressor_presence_column

    num_runs = 100
    # create some train/test splits for testing the units on equal conditions
    train_test_splits = []
    for i in range(num_runs):
        train_test_splits.append(_split(len(X), ratio=(0.8, 0.2)))

    #make runs with all units to get stable baseline
    baseline_accuracies = []
    baseline_coefs = []
    for i in range(num_runs):
        accuracy, coef = train_regressor(X, y, train_test_splits[i][0], train_test_splits[i][1])
        baseline_accuracies.append(accuracy)
        baseline_coefs.append(coef)

    baseline_coefs = np.mean(baseline_coefs, axis=0).reshape(-1)

    baseline = np.mean(baseline_accuracies)
    print('Baseline (with all units): ',baseline)

    #get majority classifier baseline
    majority_baseline = 1 - len(np.where(y == 1)[0]) / len(y)
    print('Baseline (chance): ', majority_baseline)

    #target accuracy: 95% of baseline
    target_accuracy = 0.95*baseline
    print('Target accuracy: ', target_accuracy)

    #current subset
    subset_accuracy = 0
    subset = []

    baseline_coefs = list(zip(np.arange(0, 512), baseline_coefs))
    baseline_coefs.sort(key=lambda x: x[1], reverse=True)
    print(baseline_coefs)

    while subset_accuracy < target_accuracy:
        subset.append(baseline_coefs.pop(0)[0])
        X = full_dataset.regressor_decoder_hidden_states[:,subset].reshape(-1,len(subset))
        y = full_dataset.regressor_presence_column

        accuracies_model = []
        for j in range(num_runs):
            accuracy, _ = train_regressor(X, y, train_test_splits[j][0], train_test_splits[j][1])
            accuracies_model.append(accuracy)

        subset_accuracy = np.mean(accuracies_model)
        print('units: ', subset, ' accuracy: ', subset_accuracy)

