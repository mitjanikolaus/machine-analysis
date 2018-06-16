"""
Train a diagnostic classifier to predict the existence of input sequence elements based on the hidden states of the decoder.
"""

# EXT
import torch

# PROJECT
from activations import ActivationsDataset


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

    def add_target_feature_label(self, target_feature: int, position_sensitive=-1):
        if not self.target_feature_label_added:
            # Column describing whether the target feature is present in input sequence
            self.columns += ["target_feature_present"]

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
            self.data = zip(
                self.model_inputs,
                *[getattr(activation_column) for activation_column in self.activation_columns],
                self.presence_column
            )

            self.target_feature_label_added = True


if __name__ == "__main__":
    dataset = FunctionalGroupsDataset.load("./ga_gru_1_heldout_tables.pt")
    dataset.add_target_feature_label(18, position_sensitive=2)
