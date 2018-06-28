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

# discriminator = 'guided_lstm_run_1_sample_1_heldout_tables_simple'
discriminator = 'guided_gru_run_1_sample_1_heldout_tables_simple'
data_file = discriminator + '.pt'
neurons_ordered_by_marginal = [191, 127, 389, 72, 439, 98, 438, 410, 83, 25, 210, 13, 57, 188, 355, 323, 178, 294, 203, 470, 58, 341, 480, 129, 119, 0, 49, 307, 448, 293, 71, 28, 172, 114, 471, 11, 169, 316, 148, 213, 128, 506, 109, 271, 107, 218, 298, 334, 281, 476, 241, 400, 61, 379, 14, 53, 467, 456, 412, 111, 270, 94, 136, 222, 364, 21, 187, 244, 290, 161, 287, 105, 59, 320, 176, 325, 398, 206, 216, 358, 15, 242, 17, 236, 197, 453, 252, 321, 329, 254, 414, 292, 265, 134, 69, 415, 468, 18, 229, 118, 122, 279, 10, 449, 498, 87, 103, 337, 238, 428, 366, 135, 484, 447, 437, 382, 497, 510, 375, 47, 349, 66, 133, 392, 511, 239, 299, 198, 429, 309, 23, 473, 274, 258, 158, 423, 507, 441, 342, 272, 124, 36, 30, 65, 442, 192, 492, 310, 466, 418, 33, 509, 348, 288, 221, 308, 496, 269, 284, 230, 380, 297, 50, 391, 6, 435, 7, 117, 194, 328, 276, 138, 163, 130, 202, 184, 234, 404, 131, 411, 55, 4, 201, 157, 330, 249, 387, 81, 233, 46, 8, 283, 505, 106, 141, 152, 350, 153, 417, 74, 226, 2, 376, 475, 155, 347, 273, 319, 137, 312, 326, 357, 390, 200, 32, 19, 304, 461, 318, 445, 144, 39, 351, 363, 91, 486, 338, 499, 434, 212, 215, 95, 219, 231, 422, 112, 174, 247, 503, 425, 488, 362, 132, 240, 291, 462, 262, 88, 460, 120, 121, 487, 426, 38, 457, 403, 44, 86, 90, 195, 37, 151, 92, 156, 20, 311, 378, 420, 77, 345, 204, 35, 26, 9, 243, 332, 123, 344, 259, 333, 261, 142, 78, 277, 196, 353, 162, 479, 228, 424, 477, 352, 322, 260, 175, 436, 235, 361, 248, 356, 93, 29, 478, 394, 171, 1, 96, 324, 491, 189, 483, 481, 474, 454, 75, 159, 139, 464, 365, 214, 469, 209, 343, 502, 97, 301, 143, 165, 56, 208, 340, 193, 465, 289, 452, 443, 302, 166, 73, 383, 43, 173, 371, 360, 113, 268, 68, 388, 405, 185, 278, 149, 22, 257, 227, 327, 368, 60, 354, 183, 79, 205, 305, 41, 393, 182, 490, 251, 220, 433, 280, 374, 255, 145, 245, 225, 367, 34, 489, 167, 295, 370, 459, 381, 115, 377, 237, 52, 275, 282, 27, 125, 224, 154, 190, 286, 500, 70, 48, 450, 232, 16, 432, 401, 315, 99, 331, 482, 407, 504, 444, 253, 31, 42, 303, 82, 180, 336, 455, 495, 177, 146, 3, 472, 266, 263, 54, 402, 446, 250, 199, 413, 126, 501, 168, 359, 386, 223, 63, 373, 164, 110, 406, 421, 207, 181, 12, 493, 296, 508, 89, 451, 179, 51, 100, 85, 140, 440, 335, 463, 62, 267, 5, 45, 369, 399, 419, 416, 395, 346, 494, 458, 150, 431, 408, 211, 314, 160, 285, 108, 409, 306, 427, 80, 384, 339, 430, 116, 217, 147, 40, 84, 485, 372, 397, 264, 256, 102, 300, 67, 101, 313, 104, 385, 246, 76, 317, 396, 186, 24, 170, 64]
neurons_ordered_by_singular_performance = [333, 136, 374, 242, 405, 509, 256, 345, 457, 227, 292, 342, 35, 316, 48, 499, 117, 34, 321, 248, 185, 65, 307, 235, 54, 294, 203, 85, 406, 439, 418, 59, 208, 179, 282, 67, 147, 78, 429, 188, 386, 310, 51, 273, 426, 177, 33, 275, 422, 328, 373, 26, 70, 112, 19, 126, 63, 106, 182, 189, 229, 137, 361, 145, 4, 50, 427, 210, 356, 212, 52, 420, 253, 133, 164, 508, 267, 223, 432, 447, 511, 21, 425, 412, 230, 358, 487, 459, 92, 80, 359, 119, 109, 500, 36, 129, 246, 314, 113, 471, 465, 195, 421, 349, 251, 444, 286, 448, 254, 169, 214, 424, 191, 58, 184, 56, 123, 355, 322, 205, 131, 255, 416, 96, 6, 213, 13, 378, 270, 308, 139, 98, 364, 239, 269, 318, 295, 325, 211, 86, 0, 454, 498, 363, 335, 272, 181, 497, 94, 140, 72, 473, 381, 419, 3, 477, 387, 306, 183, 299, 62, 152, 37, 397, 89, 259, 327, 236, 79, 281, 343, 68, 367, 436, 142, 16, 219, 45, 215, 122, 10, 71, 233, 130, 265, 504, 496, 190, 493, 470, 492, 348, 172, 206, 494, 451, 402, 430, 388, 110, 249, 304, 178, 196, 303, 161, 510, 241, 12, 155, 73, 450, 101, 75, 134, 481, 74, 443, 411, 396, 466, 320, 25, 357, 217, 302, 159, 352, 194, 482, 376, 283, 44, 398, 268, 490, 174, 156, 505, 7, 392, 488, 285, 125, 278, 347, 28, 30, 456, 390, 204, 313, 271, 417, 38, 480, 280, 167, 442, 312, 20, 453, 258, 403, 287, 127, 163, 46, 301, 153, 445, 1, 435, 326, 150, 128, 290, 141, 218, 111, 353, 90, 149, 173, 351, 187, 380, 344, 399, 77, 452, 431, 166, 93, 503, 64, 216, 207, 393, 479, 474, 31, 42, 84, 15, 97, 5, 160, 171, 225, 309, 501, 199, 305, 76, 102, 66, 2, 341, 124, 434, 491, 346, 467, 478, 334, 368, 484, 135, 151, 257, 263, 440, 449, 410, 291, 103, 108, 332, 250, 414, 260, 91, 116, 360, 154, 17, 462, 339, 202, 331, 23, 394, 8, 47, 400, 244, 240, 114, 247, 365, 391, 409, 121, 105, 446, 11, 407, 27, 506, 389, 100, 198, 200, 162, 288, 252, 138, 433, 39, 192, 279, 224, 118, 32, 41, 22, 362, 232, 366, 264, 507, 88, 372, 81, 165, 455, 298, 311, 107, 319, 158, 428, 300, 276, 464, 186, 143, 458, 262, 401, 476, 293, 24, 338, 289, 329, 317, 261, 296, 475, 404, 385, 460, 408, 231, 120, 9, 14, 104, 176, 53, 437, 82, 234, 297, 43, 330, 220, 18, 384, 495, 441, 228, 55, 486, 168, 157, 340, 379, 469, 350, 226, 382, 99, 375, 87, 49, 423, 237, 175, 221, 193, 463, 483, 115, 315, 95, 284, 61, 461, 277, 222, 132, 413, 60, 197, 40, 324, 337, 502, 415, 323, 201, 209, 383, 472, 468, 489, 395, 370, 369, 243, 266, 274, 180, 29, 146, 485, 170, 148, 371, 69, 245, 377, 57, 336, 354, 438, 144, 83, 238, ]
reference_mse = 0.0083264094

results = {}

for i in range(512):

    model, training_loss, validation_loss = fit_regression(
        data_directory,
        results_directory,
        data_file,
        epochs=20,
        leave_out=None,
        # feature_subset=random.sample(neurons_ordered_by_marginal[15:], 15)
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

