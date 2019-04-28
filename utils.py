# ############################################################################# #
# utils.py                                                                      #
# Author: Glenn Dawson                                                          #
# ---------------------                                                         #
# Configuration parameters and utility functions for                            #
# label-flipping-adaboost.py.                                                   #
# ############################################################################# #

import os
from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import t
import sklearn.datasets as datasets
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

seeds = [5, 7, 10, 11, 27, 42, 314, 666, 1618, 3901]  # Chosen arbitrarily
max_ensemble_size = 10000  # Show behavior up to and including this ensemble size
percent_poison = 0.1  # Fraction of training data to be poisoned
flip_original = True  # If True, directly flip data labels. Else, copy data.
plot_runs = False  # If True, plot individual run errors.
n_runs = 100  # For statistical significance
z = 1.96  # 95% confidence interval

# Save results in this local directory
savedir = '.\Results\\' + str(date.today()) + '\\2Class-10-seeds-100-runs'
if flip_original:
    savedir += '-flip-original-'
else:
    savedir += '-add-copies-'
savedir += str(int(100 * percent_poison)) + '-percent'

try:
    os.makedirs(savedir)
except FileExistsError:
    pass

def make_dataset(seed):
    return make_classification(n_samples=100000,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=2,
                               weights=None,
                               flip_y=0.0,
                               class_sep=1.0,
                               hypercube=True,
                               shift=0.0,
                               scale=1.0,
                               shuffle=True,
                               random_state=seed)

def load_dataset(dataset):
    if dataset == 'breast-cancer':
        data = pd.read_csv('.\Data\Breast-Cancer\\breast-cancer-numeric.csv',
                           index_col=0)
        labels = data.iloc[:, 0].to_numpy()
        data = data.iloc[:, 1:].to_numpy()
        return data, labels

    elif dataset == 'mushroom':
        data = pd.read_csv('.\Data\Mushroom\\agaricus-lepiota-numeric.csv',
                           index_col=0)
        labels = data.iloc[:, 0].to_numpy()
        data = data.iloc[:, 1:].to_numpy()
        return data, labels

    elif dataset == 'digits':
        digits = datasets.load_digits()
        data = digits.images.reshape((len(digits.images), -1))
        labels = digits.target
        return data, labels

    else:
        raise ValueError('Invalid dataset specified.')

def make_classifier():
    return DecisionTreeClassifier(criterion='gini',
                                  splitter='best',
                                  max_depth=1,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None,
                                  random_state=7,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  class_weight=None,
                                  presort=False)

def make_ensemble(BaseClassifier):
    print('Initializing boosted classifier...')
    return AdaBoostClassifier(BaseClassifier,
                              n_estimators=max_ensemble_size,
                              learning_rate=1.0,
                              algorithm='SAMME.R',
                              random_state=3901)

def plot_statistical_significance(test_errors, test_errors_p, seed):
    print('Calculating average errors and confidence intervals...')

    # Baseline
    errors_mean = np.mean(test_errors, axis=0)
    errors_std = np.std(test_errors, axis=0)
    confidence_upper = errors_mean + (z * (errors_std / np.sqrt(n_runs)))
    confidence_lower = errors_mean - (z * (errors_std / np.sqrt(n_runs)))

    # Poisonsed
    test_errors_p_ = np.array(test_errors_p)
    errors_mean_p = np.mean(test_errors_p_, axis=0)
    errors_std_p = np.std(test_errors_p_, axis=0)
    confidence_upper_p = errors_mean_p \
                         + (z * (errors_std_p / np.sqrt(n_runs)))
    confidence_lower_p = errors_mean_p \
                         - (z * (errors_std_p / np.sqrt(n_runs)))

    # Plot average errors and confidence intervals...
    print('Plotting statistical significance...')
    plt.figure()

    # ...for baseline ensembles
    plt.plot(range(1, len(errors_mean) + 1),
             errors_mean,
             color='C0',
             linewidth=1.75,
             label='Baseline')
    plt.plot(range(1, len(confidence_upper) + 1),
             confidence_upper,
             color='C0',
             linewidth=0.5,
             label='_nolegend_')
    plt.plot(range(1, len(confidence_lower) + 1),
             confidence_lower,
             color='C0',
             linewidth=0.5,
             label='_nolegend_')
    plt.fill_between(range(1, len(errors_mean) + 1),
                     confidence_lower,
                     confidence_upper,
                     color='C0',
                     alpha=0.5)

    # ...for poisoned ensembles
    plt.plot(range(1, len(errors_mean_p) + 1),
             errors_mean_p,
             color='C1',
             linewidth=1.75,
             label='Poisoned')
    plt.plot(range(1, len(confidence_upper_p) + 1),
             confidence_upper_p,
             color='C1',
             linewidth=0.5,
             label='_nolegend_')
    plt.plot(range(1, len(confidence_lower_p) + 1),
             confidence_lower_p,
             color='C1',
             linewidth=0.5,
             label='_nolegend_')
    plt.fill_between(range(1, len(errors_mean_p) + 1),
                     confidence_lower_p,
                     confidence_upper_p,
                     color='C1',
                     alpha=0.5)

    # Plot settings
    plt.title('Average Error for Seed ' + str(seed)
              + ' Over ' + str(n_runs) + ' Runs (95% Confidence Interval)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Error')
    plt.grid()
    plt.legend()

    # Save figure to file
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(fname=(savedir + '\seed-' + str(seed) + '\\full-plot.pdf'),
                format='pdf',
                orientation='landscape',
                bbox_inches='tight',
                dpi=1500)

def plot_statistical_significance_real(test_errors, test_errors_p, dataset,
                                       n_folds, savedir_, type='Z'):
    print('Calculating average errors and confidence intervals...')
    t_ = z
    if type == 'T':
        t_ = t.ppf(0.975, df=(n_folds - 1))

    # Baseline
    errors_mean = np.mean(test_errors, axis=0)
    errors_std = np.std(test_errors, axis=0)
    confidence_upper = errors_mean + (t_ * (errors_std / np.sqrt(n_runs)))
    confidence_lower = errors_mean - (t_ * (errors_std / np.sqrt(n_runs)))

    # Poisonsed
    test_errors_p_ = np.array(test_errors_p)
    errors_mean_p = np.mean(test_errors_p_, axis=0)
    errors_std_p = np.std(test_errors_p_, axis=0)
    confidence_upper_p = errors_mean_p \
                         + (t_ * (errors_std_p / np.sqrt(n_runs)))
    confidence_lower_p = errors_mean_p \
                         - (t_ * (errors_std_p / np.sqrt(n_runs)))

    # Plot average errors and confidence intervals...
    print('Plotting statistical significance...')
    plt.figure()

    # ...for baseline ensembles
    plt.plot(range(1, len(errors_mean) + 1),
             errors_mean,
             color='C0',
             linewidth=1.75,
             label='Baseline')
    plt.plot(range(1, len(confidence_upper) + 1),
             confidence_upper,
             color='C0',
             linewidth=0.5,
             label='_nolegend_')
    plt.plot(range(1, len(confidence_lower) + 1),
             confidence_lower,
             color='C0',
             linewidth=0.5,
             label='_nolegend_')
    plt.fill_between(range(1, len(errors_mean) + 1),
                     confidence_lower,
                     confidence_upper,
                     color='C0',
                     alpha=0.5)

    # ...for poisoned ensembles
    plt.plot(range(1, len(errors_mean_p) + 1),
             errors_mean_p,
             color='C1',
             linewidth=1.75,
             label='Poisoned')
    plt.plot(range(1, len(confidence_upper_p) + 1),
             confidence_upper_p,
             color='C1',
             linewidth=0.5,
             label='_nolegend_')
    plt.plot(range(1, len(confidence_lower_p) + 1),
             confidence_lower_p,
             color='C1',
             linewidth=0.5,
             label='_nolegend_')
    plt.fill_between(range(1, len(errors_mean_p) + 1),
                     confidence_lower_p,
                     confidence_upper_p,
                     color='C1',
                     alpha=0.5)

    # Plot settings
    plt.title('Average Error for ' + dataset + ' Dataset Over ' + str(n_folds)
              + ' Runs (95% Confidence Interval)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Error')
    plt.grid()
    plt.legend()

    # Save figure to file
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(fname=(savedir_ + '\\full-plot.pdf'),
                format='pdf',
                orientation='landscape',
                bbox_inches='tight',
                dpi=1500)