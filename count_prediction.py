import csv
import os
import random

import math
from typing import Tuple, List

import torch
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

        # self.linear = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, 1)
        # )
        self.linear = nn.Linear(input_size, 1)
        self.input_size = input_size

    def forward(self, input):
        return self.linear(input)

    def predict(self, input, leave_out_index, subset=None):

        input = input.view(1, 1, -1)
        if subset is not None: input = self._subset(input, subset)
        input = self._leave_out(input, leave_out_index)

        return self(input)

    def _leave_out(self, sample, leave_out_index):
        if leave_out_index is not None:
            if leave_out_index == 0:
                sample = sample[:, :, 1:]
            elif leave_out_index == self.input_size:
                sample = sample[:, :, :self.input_size]
            else:
                sample = torch.cat((sample[:, :, :leave_out_index], sample[:, :, leave_out_index + 1:]), dim=2)

        return sample

    def _subset(self, sample, subset):
        return sample[:, :, subset]

def fit_regression(data_directory: str, results_directory: str, data_file: str, epochs=50, leave_out=None, feature_subset:List=None):

    data = CounterDataset.load(data_directory + data_file)

    training_indices, validation_indices, test_indices = _split(len(data))

    training_data_loader = DataLoader(
        dataset=data,
        sampler=SubsetRandomSampler(training_indices)
    )

    validation_data_loader = DataLoader(
        data, sampler=SubsetRandomSampler(validation_indices)
    )

    if feature_subset is None:
        input_size = training_data_loader.dataset[0][1].size(2)
    else:
        input_size = len(feature_subset)

    if leave_out is not None: input_size -= 1

    regression = Regression(input_size)
    best_model = regression.state_dict()
    optimiser = optim.SGD(regression.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    validation_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):

        total_loss = 0
        validation_loss = 0

        for target, sample in training_data_loader:

            regression.zero_grad()


            prediction = regression.predict(sample, leave_out, feature_subset)
            loss = criterion(prediction.view(-1), target.float())

            loss.backward()
            optimiser.step()

            total_loss += loss
            # print('{:.2f}'.format(float(total_loss)))

        training_losses.append(total_loss / len(training_data_loader))

        for target, sample in validation_data_loader:
            prediction = regression.predict(sample, leave_out, feature_subset)
            validation_loss += validation_criterion(prediction.view(-1), target.float()) / len(validation_data_loader)

        validation_losses.append(validation_loss)

        if validation_loss <= min(validation_losses):
            best_model = regression.state_dict()

    plt.plot(range(len(training_losses)), [float(t) for t in training_losses], label='training loss')
    plt.plot(range(len(validation_losses)), [float(v) for v in validation_losses], label='validation loss')
    plt.legend(loc='upper right')

    plt.savefig('{}/regression_learning_curves/{}_{}.png'.format(
        results_directory,
        data_file.split('.')[0],
        leave_out
    ))
    plt.clf()

    return best_model, min(training_losses), min(validation_losses)


data_directory = 'data/decoder_counter_datasets/'
results_directory = 'results_ablation'

discriminator = 'guided_gru_run_1_sample_1_heldout_tables_simple'
data_file = discriminator + '.pt'
neurons_ordered_by_marginal = [191, 127, 389, 72, 439, 98, 438, 410, 83, 25, 210, 13, 57, 188, 355, 323, 178, 294, 203, 470, 58, 341, 480, 129, 119, 0, 49, 307, 448, 293, 71, 28, 172, 114, 471, 11, 169, 316, 148, 213, 128, 506, 109, 271, 107, 218, 298, 334, 281, 476, 241, 400, 61, 379, 14, 53, 467, 456, 412, 111, 270, 94, 136, 222, 364, 21, 187, 244, 290, 161, 287, 105, 59, 320, 176, 325, 398, 206, 216, 358, 15, 242, 17, 236, 197, 453, 252, 321, 329, 254, 414, 292, 265, 134, 69, 415, 468, 18, 229, 118, 122, 279, 10, 449, 498, 87, 103, 337, 238, 428, 366, 135, 484, 447, 437, 382, 497, 510, 375, 47, 349, 66, 133, 392, 511, 239, 299, 198, 429, 309, 23, 473, 274, 258, 158, 423, 507, 441, 342, 272, 124, 36, 30, 65, 442, 192, 492, 310, 466, 418, 33, 509, 348, 288, 221, 308, 496, 269, 284, 230, 380, 297, 50, 391, 6, 435, 7, 117, 194, 328, 276, 138, 163, 130, 202, 184, 234, 404, 131, 411, 55, 4, 201, 157, 330, 249, 387, 81, 233, 46, 8, 283, 505, 106, 141, 152, 350, 153, 417, 74, 226, 2, 376, 475, 155, 347, 273, 319, 137, 312, 326, 357, 390, 200, 32, 19, 304, 461, 318, 445, 144, 39, 351, 363, 91, 486, 338, 499, 434, 212, 215, 95, 219, 231, 422, 112, 174, 247, 503, 425, 488, 362, 132, 240, 291, 462, 262, 88, 460, 120, 121, 487, 426, 38, 457, 403, 44, 86, 90, 195, 37, 151, 92, 156, 20, 311, 378, 420, 77, 345, 204, 35, 26, 9, 243, 332, 123, 344, 259, 333, 261, 142, 78, 277, 196, 353, 162, 479, 228, 424, 477, 352, 322, 260, 175, 436, 235, 361, 248, 356, 93, 29, 478, 394, 171, 1, 96, 324, 491, 189, 483, 481, 474, 454, 75, 159, 139, 464, 365, 214, 469, 209, 343, 502, 97, 301, 143, 165, 56, 208, 340, 193, 465, 289, 452, 443, 302, 166, 73, 383, 43, 173, 371, 360, 113, 268, 68, 388, 405, 185, 278, 149, 22, 257, 227, 327, 368, 60, 354, 183, 79, 205, 305, 41, 393, 182, 490, 251, 220, 433, 280, 374, 255, 145, 245, 225, 367, 34, 489, 167, 295, 370, 459, 381, 115, 377, 237, 52, 275, 282, 27, 125, 224, 154, 190, 286, 500, 70, 48, 450, 232, 16, 432, 401, 315, 99, 331, 482, 407, 504, 444, 253, 31, 42, 303, 82, 180, 336, 455, 495, 177, 146, 3, 472, 266, 263, 54, 402, 446, 250, 199, 413, 126, 501, 168, 359, 386, 223, 63, 373, 164, 110, 406, 421, 207, 181, 12, 493, 296, 508, 89, 451, 179, 51, 100, 85, 140, 440, 335, 463, 62, 267, 5, 45, 369, 399, 419, 416, 395, 346, 494, 458, 150, 431, 408, 211, 314, 160, 285, 108, 409, 306, 427, 80, 384, 339, 430, 116, 217, 147, 40, 84, 485, 372, 397, 264, 256, 102, 300, 67, 101, 313, 104, 385, 246, 76, 317, 396, 186, 24, 170, 64]
reference_mse = 0.0083264094

results = {}

for i in range(512):

    model, training_loss, validation_loss = fit_regression(
        data_directory,
        results_directory,
        data_file,
        epochs=20,
        leave_out=None,
        feature_subset=[i]
    )

    print(float(validation_loss))



# print(results)
#
# plt.plot(range(len(results)), results.values())
# plt.savefig('ablation_plot.png')
# plt.show()

# with open('%s/regression_evaluation.csv' % results_directory, 'a') as evaluation:
#
#     writer = csv.writer(evaluation)
#
#     for leave_out in range(512):
#
#         if discriminator + '.png' in os.listdir('%s/regression_learning_curves/' % results_directory): continue
#
#         model, training_loss, validation_loss = fit_regression(data_directory, results_directory, data_file, epochs=20,
#                                                                leave_out=leave_out)
#         print('Leaving out {}, validation loss {:.5f}'.format(leave_out, validation_loss))
#
#         writer.writerow([leave_out, float(training_loss), float(validation_loss)])
#
#         torch.save(model, '{}/models/{}.pt'.format(results_directory, leave_out))

