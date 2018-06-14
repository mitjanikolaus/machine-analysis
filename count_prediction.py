import csv
import os
import random

import math
from typing import Tuple, List

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from activations import CounterDataset

import matplotlib.pyplot as plt


def _split(length: int, ratio=(0.9, 0.05, 0.05), seed=1) -> Tuple[List[int], List[int], List[int]]:

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


class Regression(nn.Module):

    def __init__(self, input_size):
        super(Regression, self).__init__()

        hidden_size = 30

        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        return self.linear(input)

    def predict(self, input):
        return self(input)


def fit_regression(directory: str, file: str, epochs=50):

    print(file)

    data = CounterDataset.load(directory + file)

    training_indices, validation_indices, test_indices = _split(len(data))

    training_data_loader = DataLoader(
        dataset=data,
        sampler=SubsetRandomSampler(training_indices)
    )

    validation_data_loader = DataLoader(
        data, sampler=SubsetRandomSampler(validation_indices)
    )

    regression = Regression(512)
    optimiser = optim.SGD(regression.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    validation_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):

        total_loss = 0
        validation_loss = 0

        for target, sample in training_data_loader:

            optimiser.zero_grad()
            regression.zero_grad()

            prediction = regression.predict(sample.view(-1))
            loss = criterion(prediction, target.float())

            loss.backward()
            optimiser.step()

            total_loss += loss
            # print('{:.2f}'.format(float(total_loss)))

        training_losses.append(total_loss / len(training_data_loader))

        for target, sample in validation_data_loader:
            prediction = regression(sample.view(-1))
            validation_loss += validation_criterion(prediction, target.float())

        validation_losses.append(validation_loss / len(validation_data_loader))

    plt.plot(range(len(training_losses)), [float(t) for t in training_losses], label='training loss')
    plt.plot(range(len(validation_losses)), [float(v) for v in validation_losses], label='validation loss')
    plt.legend(loc='upper right')

    plt.savefig('plots/regression_learning_curves/{}.png'.format(file.split('.')[0]))
    plt.clf()

    return min(training_losses), min(validation_losses)


data_directory = 'data/counter_datasets/'

with open('plots/regression_evaluation.csv', 'a') as evaluation:

    writer = csv.writer(evaluation)

    for file in os.listdir(data_directory):

        if file.split('.')[0] + '.png' in os.listdir('plots/regression_learning_curves/'): continue

        if file.endswith('.pt'):
            training_loss, validation_loss = fit_regression(data_directory, file, epochs=50)
            writer.writerow([file.split('.')[0], float(training_loss), float(validation_loss)])

