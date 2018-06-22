"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# STD
import random
import math

# EXT
from sklearn.linear_model import LogisticRegression
from collections import Counter
import numpy as np

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
        self.regressor_label_column = []
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

    def add_dataset_for_regressor(self, target_features: [], target_activations: str, target_position=-1):
        """
        Add a binary label to every instance in the data set, telling whether a target feature is present in the input
        sequence (and if it is at a specified position given that position_sensitive > -1).
        """
        if not self.target_feature_label_added:
            # Column describing whether the target feature is present in input sequence
            self.columns = [target_activations, "target_feature_present"]

            regressor_inputs = []
            regressor_class_labels = []

            for model_input, target_activations_timesteps in zip(self.model_inputs, getattr(self, target_activations)):
                    for ts, target_activation in enumerate(target_activations_timesteps):
                        #disregarding position
                        if target_position == -1:
                            # Target feature in input: label 1
                            if model_input.flatten()[ts] in target_features:
                                regressor_inputs.append(target_activation)
                                label = target_features.index(model_input.flatten()[ts])
                                regressor_class_labels.append(label)

                        #regarding position
                        else:
                            if ts == target_position and model_input.flatten()[target_position] in target_features:
                                regressor_inputs.append(target_activation)
                                label = target_features.index(model_input.flatten()[target_position])
                                regressor_class_labels.append(label)

            # Overwrite data using the new class label column
            self.regressor_inputs = np.array(regressor_inputs)
            self.regressor_label_column = np.array(regressor_class_labels)


            self.target_feature_label_added = True  # Only allow this logic to be called once


def perform_ablation_study(activations_dataset_path, target_features, target_position, input_for_prediction, num_runs=1000, train_test_split=(0.9, 0.1),
                           target_accuracy_cut=0.95):
    """
    Perform an ablation study by stepwise adding units to a subset until target accuracy is reached. The units are
    chosen based on the weights they were assigned in a logistic regressor

    :param activations_dataset_path: path to dataset of activations
    :param target_feature: feature that should be predicted
    :param num_runs: number of runs to be performed to get average accuracies
    :param train_test_split: tuple denoting percentage of data in training and test set
    :param target_accuracy_cut: accuracy that should be reached to converge (target_accuracy = target_accuracy_cut * baseline_accuracy)
    :return: list denoting the subset of units that are needed to reach the target accuracy
    """
    # Load data and split into sets
    full_dataset = FunctionalGroupsDataset.load(activations_dataset_path, convert_to_numpy=True)
    full_dataset.add_dataset_for_regressor(
        target_features=target_features, target_activations=input_for_prediction,target_position=target_position
    )

    def train_regressor(X, y, training_indices, test_indices):
        #TODO multi_class = ovr or multinomial
        #TODO solver? max_iter?
        regressor = LogisticRegression(n_jobs=-1, verbose=0, multi_class='multinomial',solver='saga', max_iter=100)
        regressor.fit(X[training_indices], y[training_indices])

        return regressor.score(X[test_indices], y[test_indices]), regressor.coef_

    # create input and target for regressor
    X = full_dataset.regressor_inputs
    y = full_dataset.regressor_label_column

    # majority classifier baseline
    c = Counter(y)
    majority_baseline = c.most_common(1)[0][1] / len(y)
    print('Baseline (majority): ', majority_baseline)

    # create some train/test splits for testing the accuracies on equal conditions
    train_test_splits = []
    for i in range(num_runs):
        train_test_splits.append(_split(len(X), ratio=train_test_split))

    # make runs with all units to get stable baseline
    baseline_accuracies = []
    baseline_coefs = []
    for i in range(num_runs):
        accuracy, coef = train_regressor(X, y, train_test_splits[i][0], train_test_splits[i][1])
        baseline_accuracies.append(accuracy)
        baseline_coefs.append(coef)

    # create average over the regressor coefficients to know which units are important
    baseline_coefs = np.abs(np.mean(baseline_coefs, axis=0).reshape(-1))
    baseline_coefs = list(zip(np.arange(0, 512), baseline_coefs))
    baseline_coefs.sort(key=lambda x: x[1], reverse=True)

    # baseline when using all units
    baseline = np.mean(baseline_accuracies)
    print('Baseline (with all units): ', baseline)

    # target accuracy: depending on baseline
    target_accuracy = target_accuracy_cut * baseline
    print('Target accuracy: ', target_accuracy)

    # current subset of units to test
    subset_accuracy = 0
    subset = []
    while subset_accuracy < target_accuracy:
        # add the next unit with highest weight in regressor
        subset.append(baseline_coefs.pop(0)[0])
        X = full_dataset.regressor_inputs[:, subset].reshape(-1, len(subset))
        y = full_dataset.regressor_label_column

        # perform some runs to get average accuracy
        accuracies_model = []
        for j in range(num_runs):
            accuracy, _ = train_regressor(X, y, train_test_splits[j][0], train_test_splits[j][1])
            accuracies_model.append(accuracy)

        subset_accuracy = np.mean(accuracies_model)
        print('units: ', subset, ' accuracy: ', subset_accuracy)

    subset.sort()
    print('units (sorted): ', subset, ' accuracy: ', subset_accuracy)

    return subset, subset_accuracy

#Hidden units
ACTIVATIONS_HIDDEN_UNITS_DECODER = 'hidden_activations_decoder'
ACTIVATIONS_HIDDEN_UNITS_ENCODER = 'hidden_activations_encoder'

#GRU Gates
ACTIVATIONS_GRU_INPUT_GATE_DECODER = 'input_gate_activations_decoder'
ACTIVATIONS_GRU_INPUT_GATE_ENCODER = 'input_gate_activations_encoder'
ACTIVATIONS_GRU_RESET_GATE_DECODER = 'reset_gate_activations_decoder'
ACTIVATIONS_GRU_RESET_GATE_ENCODER = 'reset_gate_activations_encoder'

#LSTM Gates
ACTIVATIONS_LSTM_INPUT_GATE_DECODER = 'input_gate_activations_decoder'
ACTIVATIONS_LSTM_INPUT_GATE_ENCODER = 'input_gate_activations_encoder'
ACTIVATIONS_LSTM_FORGET_GATE_DECODER = 'forget_gate_activations_decoder'
ACTIVATIONS_LSTM_FORGET_GATE_ENCODER = 'forget_gate_activations_encoder'
ACTIVATIONS_LSTM_OUTPUT_GATE_DECODER = 'output_gate_activations_decoder'
ACTIVATIONS_LSTM_OUTPUT_GATE_ENCDODER = 'output_gate_activations_encoder'

if __name__ == "__main__":
    # Input vocabulary indices (
    # '<pad>: 1
    # 011: 10
    # .: 2
    # t1: 3
    # 010: 14
    # 101: 9
    # 000: 13
    # t2: 4
    # 110: 12
    # t4: 5
    # t8: 18
    # t5: 6
    # < unk >: 0
    # t3: 7
    # 111: 15
    # 100: 11
    # 001: 16
    # t6: 8
    # t7: 17

    target_features = [3,4,5,6,7,8,17,18]  # tables = [3,4,5,6,7,8,17,18] # inputs = [10,14,9,13,12,15,11,16]
    target_position = -1 # either a specific timestep or -1 for disregarding the timestep
    activations_dataset_path = "./data/guided_lstm_1_all.pt"
    print(activations_dataset_path)

    input_for_prediction = ACTIVATIONS_LSTM_OUTPUT_GATE_DECODER

    # number of runs to get average of classifier weights
    num_runs = 5

    subset, subset_accuracy = perform_ablation_study(activations_dataset_path, target_features, target_position, input_for_prediction, num_runs=num_runs)

    print(len(subset))


