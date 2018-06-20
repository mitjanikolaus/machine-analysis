"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# STD
import random
import math
import warnings

# EXT
from sklearn.linear_model import LogisticRegression
import numpy as np
import shap

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
            regressor_decoder_hidden_states = positive_decoder_hidden_states + negative_decoder_hidden_states[:num_positive]
            regressor_class_labels = [1] * num_positive + [0] * num_positive

            # Overwrite data using the new class label column
            self.regressor_decoder_hidden_states = np.array(regressor_decoder_hidden_states)
            self.regressor_presence_column = np.array(regressor_class_labels)

            self.length = 2 * num_positive

            self.target_feature_label_added = True  # Only allow this logic to be called once


def get_neuron_shapley_values(regressor, X_train, X_test, k=50, nsamples=100):
    summary = shap.kmeans(X_train, k=k)
    explainer = shap.KernelExplainer(regressor.predict_proba, summary)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapley_values = explainer.shap_values(X_test, nsamples=nsamples)
        shapley_values = shapley_values[0]
        shapley_values = shapley_values[:, :X_test.shape[1]]

    return shapley_values


if __name__ == "__main__":
    num_models = 10
    target_feature = 3  # t1 = 3
    shapley_values = None

    # Load data and split into sets
    full_dataset = FunctionalGroupsDataset.load("./guided_gru_1_train.pt", convert_to_numpy=True)
    full_dataset.add_dataset_for_regressor(
        target_feature=target_feature, target_activations="hidden_activations_decoder"
    )

    for i in range(num_models):
        # create a new random split for each model
        training_indices, test_indices = _split(len(full_dataset), ratio=(0.9, 0.1))

        X_train = full_dataset.regressor_decoder_hidden_states[training_indices]
        y_train = full_dataset.regressor_presence_column[training_indices]
        X_test = full_dataset.regressor_decoder_hidden_states[test_indices]
        y_test = full_dataset.regressor_presence_column[test_indices]

        regressor = LogisticRegression()
        regressor.fit(X_train, y_train)
        print('Accuracy:', regressor.score(X_test, y_test))

        # Collect shapley values
        current_shapley_values = get_neuron_shapley_values(regressor, X_train, X_test)
        shapley_values = current_shapley_values if shapley_values is None else \
            np.concatenate((shapley_values, current_shapley_values), axis=0)

    # Summarize
    summarized_shapley_values = shapley_values.sum(axis=0)
    summarized_shapley_values /= summarized_shapley_values.sum(axis=0)
    contributing_neurons = np.where[summarized_shapley_values > summarized_shapley_values.mean() + 2 * summarized_shapley_values.std()][0]
    print(contributing_neurons)
    a = 3


