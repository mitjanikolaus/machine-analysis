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
from visualization import plot_multiple_model_weights


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


def get_neuron_shapley_values(regressor, X_train, X_test, k=100, nsamples=100):
    summary = shap.kmeans(X_train, k=k)
    med = np.median(X_train, axis=0).reshape(1, -1)
    num_features = X_test.shape[1]
    explainer = shap.KernelExplainer(regressor.predict_proba, summary)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapley_values = explainer.shap_values(X_test, nsamples=nsamples)
        shapley_values = shapley_values[0][:, :num_features]

    return shapley_values


if __name__ == "__main__":
    num_models = 10
    target_feature = 3  # t1 = 3
    shapley_values = None
    regressors = []

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
        regressors.append(regressor)
        print('Accuracy:', regressor.score(X_test, y_test))

        # Collect shapley values
        current_shapley_values = get_neuron_shapley_values(regressor, X_train, X_test)
        shapley_values = current_shapley_values if shapley_values is None else \
            np.concatenate((shapley_values, current_shapley_values), axis=0)

    # Summarize
    summarized_shapley_values = shapley_values.sum(axis=0)
    summarized_shapley_values /= summarized_shapley_values.sum(axis=0)
    contributing_neurons = np.where(np.abs(summarized_shapley_values) > summarized_shapley_values.mean() + 2 * summarized_shapley_values.std())[0]
    print(contributing_neurons)
    # Zero background model
    # [ 13  44  92 106 112 178 182 193 250 302 307 321 328 371 382 404 426 436 439 441 444 459 473]
    # [ 8  13  44  92 106 112 136 151 165 168 178 182 188 202 268 286 302 307 321 328 353 382 404 426 439 441 444 459 466 473 494]

    # Median background model
    # [ 13  25  59  65 106 151 178 193 211 227 268 273 274 298 302 321 382 405 426 435 436 439 444 450]
    # [ 13  25  65  72 106 151 176 178 183 193 211 227 298 302 321 353 382 426 436 439 441 444 473]

    # K-means background model
    # [ 5  11  16  19  25  40  45  49  50  53  66 106 117 135 143 163 191 209 322 331 357 398 416 431 439 472 494 501]
    # [  3  12  13  14  15  57  63  65  77  78  86 118 120 123 134 146 200 222 232 241 249 268 293 310 385 434 439 443 461 502 507]

    model_weights = [regressor.coef_ for regressor in regressors]
    plot_multiple_model_weights(model_weights)

