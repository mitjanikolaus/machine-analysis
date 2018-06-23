
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

from models.analysable_seq2seq import AnalysableSeq2seq
from activations import ActivationsDataset

def run_analysis(rnn_type, similar_input_criterium, other_input_criterium, models_to_evaluate, model_part_to_evaluate, compare, print_outliers=True, legend_labels=('similar input','other input'), legend_location=(2,2)):

    distances_baseline_similar_all_models = []
    distances_baseline_other_all_models = []
    distances_guided_similar_all_models = []
    distances_guided_other_all_models = []

    for model_id in models_to_evaluate:
        for i, model_type in enumerate(['baseline', 'guided']):
            print('evaluating ',model_type, model_id)

            activations_dataset_path = 'data/' + model_type + '_' + rnn_type + '_'+str(model_id)+'_all.pt'

            samples_group_similar, samples_group_other = prepare_test_data(activations_dataset_path, similar_input_criterium, other_input_criterium, model_part_to_evaluate)

            distances_similar, distances_other = run_models_and_evaluate(samples_group_similar, samples_group_other, compare, print_outliers)

            if model_type == 'baseline':
                if distances_baseline_similar_all_models == []:
                    distances_baseline_similar_all_models = [[] for n in range(len(distances_similar))]
                    distances_baseline_other_all_models = [[] for n in range(len(distances_other))]
                else:
                    for ts in range(len(distances_similar)):
                        distances_baseline_similar_all_models[ts].extend(distances_similar[ts])
                        distances_baseline_other_all_models[ts].extend(distances_other[ts])
            if model_type == 'guided':
                if distances_guided_similar_all_models == []:
                    distances_guided_similar_all_models = [[] for n in range(len(distances_similar))]
                    distances_guided_other_all_models = [[] for n in range(len(distances_other))]
                else:
                    for ts in range(len(distances_similar)):
                        distances_guided_similar_all_models[ts].extend(distances_similar[ts])
                        distances_guided_other_all_models[ts].extend(distances_other[ts])

    timesteps = np.arange(len(distances_baseline_similar_all_models))

    #perform t-tests
    print('t-test results: ')
    for ts in timesteps:
        print('Timestep: ',ts)
        print('Baseline: ')
        print(ttest_rel(distances_baseline_similar_all_models[ts], np.random.choice(distances_baseline_other_all_models[ts],size=len(distances_baseline_similar_all_models[ts]), replace=False))[1])
        print('Guided: ')
        print(ttest_rel(distances_guided_similar_all_models[ts],np.random.choice(distances_guided_other_all_models[ts], size=len(distances_guided_similar_all_models[ts]), replace=False))[1])

    width = 0.35  # the width of the bars
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    plt.suptitle(model_part_to_evaluate)

    ax1.set_title('Baseline Model')

    rects1_baseline = ax1.bar(timesteps, [np.mean(i) for i in distances_baseline_similar_all_models], width, color='tab:blue',yerr=[np.std(i) for i in distances_baseline_similar_all_models])
    rects2_baseline = ax1.bar(timesteps + width, [np.mean(i) for i in distances_baseline_other_all_models], width, color='tab:blue', alpha=0.5,yerr=[np.std(i) for i in distances_baseline_other_all_models])
    ax1.set_ylabel('Mean distances ')

    ax2.set_title('Guided Attention Model')
    rects1_guided = ax2.bar(timesteps, [np.mean(i) for i in distances_guided_similar_all_models], width, color='tab:orange',yerr=[np.std(i) for i in distances_guided_similar_all_models])
    rects2_guided = ax2.bar(timesteps + width, [np.mean(i) for i in distances_guided_other_all_models], width, color='tab:orange', alpha=0.5,yerr=[np.std(i) for i in distances_guided_other_all_models])
    ax2.set_ylabel('Mean distances')

    if compare == COMPARE_SAMPLES:
        plt.xticks(timesteps + width / 2, ['t={}'.format(i) for i in timesteps])
    if compare == COMPARE_TIMESTEPS:
        ax2.set_xticks(timesteps + width / 2)
        ax2.set_xticklabels(['t{} -> t{}'.format(i, i+1) for i in timesteps])

    ax1.legend((rects1_baseline[0], rects2_baseline[0]), legend_labels, loc=legend_location[0])
    ax2.legend((rects1_guided[0], rects2_guided[0]), legend_labels, loc=legend_location[1])

    plt.show()

def run_models_and_evaluate(activation_data_similar, activation_data_other, compare, print_outliers=True):
    if compare == COMPARE_SAMPLES:
        distances_similar_each_timestep = calculate_distances_among_samples(activation_data_similar)
        distances_different_each_timestep = calculate_distances_among_samples(activation_data_other)
    elif compare == COMPARE_TIMESTEPS:
        distances_similar_each_timestep = calculate_distances_among_timesteps(activation_data_similar)
        distances_different_each_timestep = calculate_distances_among_timesteps(activation_data_other)
    else:
        raise ValueError("compare must be either COMPARE_TIMESTEPS or COMPARE_SAMPLES")

    results_similar_each_timestep = []
    results_different_each_timestep = []

    for ts in range(len(distances_similar_each_timestep)):
        if print_outliers:
            mean_distances_similar_per_cell = np.mean(distances_similar_each_timestep[ts], axis=0)
            mean_distances_different_per_cell = np.mean(distances_different_each_timestep[ts], axis=0)

            outliers_similar_low_change = np.where(mean_distances_similar_per_cell < (np.mean(mean_distances_similar_per_cell) - 2 * np.std(mean_distances_similar_per_cell)))[0]
            outliers_different_low_change = np.where(mean_distances_different_per_cell < (np.mean(mean_distances_different_per_cell) - 2 * np.std(mean_distances_different_per_cell)))[0]
            outliers_similar_high_change = np.where(mean_distances_similar_per_cell > (np.mean(mean_distances_similar_per_cell) + 2 * np.std(mean_distances_similar_per_cell)))[0]
            outliers_different_high_change = np.where(mean_distances_different_per_cell > (np.mean(mean_distances_different_per_cell) + 2 * np.std(mean_distances_different_per_cell)))[0]

            relevant_outliers_low_change = list(set(outliers_similar_low_change) - set(outliers_different_low_change))
            relevant_outliers_high_change = list(set(outliers_similar_high_change) - set(outliers_different_high_change))
            relevant_outliers_low_change.sort()
            relevant_outliers_high_change.sort()
            print('Relevant outliers timestep', ts, 'low change: ', relevant_outliers_low_change)
            print('Relevant outliers timestep', ts, 'high change: ', relevant_outliers_high_change)


        results_similar_each_timestep.append(np.sqrt(np.sum(distances_similar_each_timestep[ts],axis=1)))
        results_different_each_timestep.append(np.sqrt(np.sum(distances_different_each_timestep[ts],axis=1)))

    return results_similar_each_timestep, results_different_each_timestep

def calculate_distances_among_samples(activation_data):
        distances = []
        for i, input_sample in enumerate(activation_data):
            for ts, input_sample_timestep in enumerate(input_sample):
                if len(distances) <= ts:
                    distances.append([])
                for j, input_sample_2 in enumerate(activation_data):
                    if not j == i:
                        if len(input_sample_2) >= len(input_sample):
                            single_distances = (np.square(np.subtract(input_sample_timestep, input_sample_2[ts])))
                            distances[ts].append(single_distances)
        return distances

def calculate_distances_among_timesteps(activation_data):
    distances = []
    for i, input_sample in enumerate(activation_data):
        for ts, input_sample_timestep in enumerate(input_sample):
            if ts < len(input_sample)-1:
                if len(distances) <= ts:
                    distances.append([])
                single_distances = (np.square(np.subtract(input_sample_timestep, input_sample[ts+1])))
                distances[ts].append(single_distances)
    return distances

def prepare_test_data(activations_dataset_path, similar_input_criterium, other_input_criterium, model_part_to_evaluate):
    samples_group_similar = []  # all samples that start with t1
    samples_group_other = []  # all other samples

    activation_dataset = ActivationsDataset.load(activations_dataset_path, convert_to_numpy=True)

    for input, activation_data in zip(activation_dataset.model_inputs, getattr(activation_dataset, model_part_to_evaluate)):
        if similar_input_criterium(input[0]):
            samples_group_similar.append(activation_data)
        if other_input_criterium(input[0]):
            samples_group_other.append(activation_data)

    return samples_group_similar, samples_group_other

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

COMPARE_SAMPLES = 'compare_among_different_samples'
COMPARE_TIMESTEPS = 'compare_among_different_timesteps'


if __name__ == "__main__":
    models_to_evaluate = [1,2,3,4,5]  # 1,2,3,4,5 (which models from the zoo to take)
    print_outliers = True

    rnn_type = 'gru' #gru or lstm

    model_part_to_evaluate = ACTIVATIONS_HIDDEN_UNITS_DECODER

    # either compare among different samples or compare among different timesteps
    compare = COMPARE_SAMPLES # COMPARE_SAMPLES or COMPARE_TIMESTEPS

    # load a checkpoint to get input vocab
    checkpoint_path = '../machine-zoo/baseline/' + rnn_type + '/' + str(models_to_evaluate[0]) + '/'
    checkpoint = AnalysableSeq2seq.load(checkpoint_path)
    input_vocab = checkpoint.input_vocab

    # criterium for filtering inputs
    def similar_input_criterium(sample):
        return input_vocab.itos[sample[0]] == '000' and input_vocab.itos[sample[1]] == 't1' #and input_vocab.itos[sample[2]] == 't1' #

    #c riterium for baseline to compare to
    def other_input_criterium(sample):
        # return just True to get a baseline of all samples
        return True and input_vocab.itos[sample[0]] == '000'

    legend_labels = ('Input: XXX t1 X','Input: XXX X X')
    legend_location = (2,4) #location of legend for (baseline, guided) graphs
    run_analysis(rnn_type, similar_input_criterium, other_input_criterium, models_to_evaluate, model_part_to_evaluate, compare, print_outliers=print_outliers, legend_labels=legend_labels, legend_location=legend_location)
