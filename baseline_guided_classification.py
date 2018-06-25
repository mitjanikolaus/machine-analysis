import os
import numpy as np
from sklearn.linear_model import LogisticRegression

from activations import CounterDataset


def keywords_in_dataset(dataset_path: str, keywords: []) -> bool:
    return all([keyword in dataset_path for keyword in keywords])

def data_from_counter(source_dataset: CounterDataset, label, filter):

    data = np.empty((0, 512))
    labels = []

    for target, item in source_dataset:
        if filter(target):
            data = np.concatenate((data, item.view(1,-1).numpy()))
            labels.append(label)

    return data, labels

baseline_paths = filter(lambda dataset: keywords_in_dataset(dataset, ['baseline', 'simple']), os.listdir('data/encoder_counter_datasets/'))
guided_paths = filter(lambda dataset: keywords_in_dataset(dataset, ['guided', 'simple']), os.listdir('data/encoder_counter_datasets/'))


baseline_dataset = np.empty((0, 512))
baseline_labels = []

guided_dataset = np.empty((0, 512))
guided_labels = []

for dataset_path in baseline_paths:

    dataset = CounterDataset.load('data/encoder_counter_datasets/' + dataset_path)
    data, labels = data_from_counter(dataset, 0, lambda target: target == 2)
    baseline_dataset = np.concatenate((baseline_dataset, data))
    baseline_labels.extend(labels)

for dataset_path in guided_paths:

    dataset = CounterDataset.load('data/encoder_counter_datasets/' + dataset_path)
    data, labels = data_from_counter(dataset, 1, lambda target: target == 2)
    guided_dataset = np.concatenate((guided_dataset, data))
    guided_labels.extend(labels)


full_data, full_labels = np.concatenate((baseline_dataset, guided_dataset)), np.array(baseline_labels + guided_labels)

number_of_items = len(full_labels)

labeled_data = np.concatenate((full_data, np.reshape(full_labels, (-1, 1))), 1)
np.random.shuffle(labeled_data)

labeled_training_data, labeled_test_data = np.split(labeled_data, [int(number_of_items*0.9)])

training_data = labeled_training_data[:, :-1]
test_data = labeled_test_data[:, :-1]
training_labels = labeled_training_data[:, -1]
test_labels = labeled_test_data[:, -1]

def get_accuracy(predictions, labels) -> float:
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(predictions)

logistic_regression = LogisticRegression()
logistic_regression.fit(
    training_data, training_labels
)

accuracy = get_accuracy(logistic_regression.predict(test_data), test_labels)

print('Accuracy: ', accuracy)


print('')

