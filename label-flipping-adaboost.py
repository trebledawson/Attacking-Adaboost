# ############################################################################# #
# label-flipping-adaboost.py                                                    #
# Author: Glenn Dawson                                                          #
# ---------------------------                                                   #
# This file attempts to poison the Adaboost algorithm by flipping the labels of #
# a certain percentage of the training data. There are two possible attack      #
# schemas:                                                                      #
#  * First, the attacker is assumed to have the ability to directly flip the    #
#    labels of the genuine data in the training dataset.                        #
#  * Second, the attack is not assumed to have this ability, but can instead    #
#    insert copies of the genuine data, with flipped labels, into the training  #
#    dataset.                                                                   #
# These two schemas may be toggled by flipping the "flip-original" flag; if     #
# True, then the first schema is chosen; otherwise the second schema is chosen. #
#                                                                               #
# Experiments are carried out on a number of classification problems equal to   #
# len(seeds). For each problem, 100,000 samples are generated. Over n_runs      #
# iterations, an Adaboost classifier built on decision trees is trained, first  #
# on the baseline, unpoisoned dataset, and then on the dataset poisoned by      #
# label-flipping. The average error over n_runs is reported, along with a 95%   #
# confidence interval.                                                          #
# ############################################################################# #

import os
import time
from datetime import date
from copy import deepcopy
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

seeds = [5, 7, 10, 11, 27, 42, 314, 666, 1618, 3901]  # Chosen arbitrarily
max_ensemble_size = 10000  # Show behavior up to and including this ensemble size
flip_original = True  # If True, directly flip data labels. Else, copy data.
plot_runs = False  # If True, plot individual run errors.
n_runs = 100  # For statistical significance
z = 1.96  # 95% confidence interval

# Save results in this local directory
savedir = '.\Results\\' + str(date.today()) + '\\2Class-10-seeds-100-runs'
try:
    os.makedirs(savedir)
except FileExistsError:
    pass

def main():
    start = time.time()
    for seed in seeds:
        less_than_max = False
        test_errors_ = []
        test_errors_p_ = []
        print('Generating data...')
        data, labels = make_classification(n_samples=100000,
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

        skf = StratifiedKFold(n_splits=n_runs)
        run = 1
        for _, fold in skf.split(X=data, y=labels):
            print('Seed:', seed, '| Run:', run)
            run += 1

            print('Splitting data...')
            d_fold = data[fold, :]
            y_fold = labels[fold]
            d_train, d_test, y_train, y_test = tts(d_fold, y_fold,
                                                   test_size=0.3,
                                                   shuffle=True,
                                                   stratify=None)

            print('Initializing ensemble base classifier...')
            BaseClassifier = make_classifier()

            print('Generating poisoned dataset...')
            flip_idx = np.random.choice(range(len(y_train)),
                                        size=int(0.1*len(y_train)),
                                        replace=False)

            if flip_original:
                d_train_p = d_train
                y_train_p = deepcopy(y_train)
                for idx in flip_idx:
                    if y_train_p[idx] == 0:
                        y_train_p[idx] = 1
                    else:
                        y_train_p[idx] = 0
            else:
                d_p = [d_train[idx, :] for idx in flip_idx]
                y_p = [1 if y_train[idx] == 0 else 0 for idx in flip_idx]
                d_train_p = np.vstack((d_train, d_p))
                y_train_p = np.hstack((y_train, y_p))

            # Train and test on baseline data
            print('Fitting model on training data...')
            ensemble = make_ensemble(BaseClassifier)
            ensemble.fit(d_train, y_train)

            print('Predicting on test data...')
            test_errors = [1. - accuracy_score(y_test, p) for p in
                           ensemble.staged_predict(d_test)]

            ensemble_errors = ensemble.estimator_errors_[:len(ensemble)]

            # Train and test on poisoned data
            print('Fitting model on poisoned training data...')
            ensemble_p = make_ensemble(BaseClassifier)
            ensemble_p.fit(d_train_p, y_train_p)

            print('Predicting on test data...')
            test_errors_p = [1. - accuracy_score(y_test, p) for p in
                             ensemble_p.staged_predict(d_test)]

            ensemble_errors_p = ensemble_p.estimator_errors_[:len(ensemble_p)]

            # Store individual run errors for statistical analysis
            test_errors_.append(test_errors)
            test_errors_p_.append(test_errors_p)

            print('Baseline error:', test_errors[-1],
                  '| # Trees:', len(ensemble))
            print('Poisoned error:', test_errors_p[-1],
                  '| # Trees:', len(ensemble_p))

            # Plot individual run errors
            if plot_runs:
                print('Plotting...')
                plt.figure()
                plt.suptitle('Testing on AdaBoost')
                plt.subplot(311)
                plt.plot(range(1, len(test_errors) + 1),
                         test_errors,
                         label='Baseline')
                plt.plot(range(1, len(test_errors_p) + 1),
                         test_errors_p,
                         label='Poisoned')
                plt.ylabel('Test Error')
                plt.grid()
                plt.legend()

                plt.subplot(312)
                plt.plot(range(1, len(ensemble_errors) + 1),
                         ensemble_errors,
                         c='black', label='Baseline')
                plt.ylabel('Classifier Error')
                plt.grid()
                plt.legend()

                plt.subplot(313)
                plt.plot(range(1, len(ensemble_errors_p) + 1),
                         ensemble_errors_p,
                         c='black', label='Poisoned')
                plt.xlabel('Number of Trees')
                plt.ylabel('Classifier Error')
                plt.grid()
                plt.legend()

            # Check if Adaboost algorithm terminated early (may cause issues
            # in statistical analysis)
            if len(ensemble) < max_ensemble_size or \
                            len(ensemble_p) < max_ensemble_size:
                less_than_max = True
            if less_than_max:
                print('Warning: Ensemble created with fewer than',
                      max_ensemble_size, 'trees.')
            print('---------------------------')

        # Statistical analysis over all runs...
        print('Calculating average errors and confidence intervals...')

        # ...for baseline ensembles
        test_errors_ = np.array(test_errors_)
        errors_mean = np.mean(test_errors_, axis=0)
        errors_std = np.std(test_errors_, axis=0)
        confidence_upper = errors_mean + (z * (errors_std / np.sqrt(n_runs)))
        confidence_lower = errors_mean - (z * (errors_std / np.sqrt(n_runs)))

        # ...for poisoned ensembles
        test_errors_p_ = np.array(test_errors_p_)
        errors_mean_p = np.mean(test_errors_p_, axis=0)
        errors_std_p = np.std(test_errors_p_, axis=0)
        confidence_upper_p = errors_mean_p \
                             + (z * (errors_std_p / np.sqrt(n_runs)))
        confidence_lower_p = errors_mean_p \
                             - (z * (errors_std_p / np.sqrt(n_runs)))

        # Plot average errors and confidence intervals...
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
        fig.savefig(fname=(savedir + '\\seed-' + str(seed) + '.pdf'),
                    format='pdf',
                    orientation='landscape',
                    bbox_inches='tight',
                    dpi=1500)

    print('Runtime:', time.time() - start, 'seconds.')
    plt.show()
    print('Done.')

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

if __name__ == '__main__':
    main()