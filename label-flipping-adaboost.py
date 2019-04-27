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
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils

# Save results in this local directory
try:
    os.makedirs(utils.savedir)
except FileExistsError:
    pass

def main():
    start = time.time()
    for seed in utils.seeds:
        less_than_max = False
        test_errors_ = []
        test_errors_p_ = []
        print('Generating data...')
        data, labels = utils.make_dataset(seed)

        skf = StratifiedKFold(n_splits=utils.n_runs)
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
            BaseClassifier = utils.make_classifier()

            print('Generating poisoned dataset...')
            flip_idx = np.random.choice(range(len(y_train)),
                                        size=int(utils.percent_poison
                                                 * len(y_train)),
                                        replace=False)

            if utils.flip_original:
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
            ensemble = utils.make_ensemble(BaseClassifier)
            ensemble.fit(d_train, y_train)

            print('Predicting on test data...')
            test_errors = [1. - accuracy_score(y_test, pred) for pred in
                           ensemble.staged_predict(d_test)]

            ensemble_errors = ensemble.estimator_errors_[:len(ensemble)]

            # Train and test on poisoned data
            print('Fitting model on poisoned training data...')
            ensemble_p = utils.make_ensemble(BaseClassifier)
            ensemble_p.fit(d_train_p, y_train_p)

            print('Predicting on test data...')
            test_errors_p = [1. - accuracy_score(y_test, pred) for pred in
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
            if utils.plot_runs:
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
            if len(ensemble) < utils.max_ensemble_size or \
                            len(ensemble_p) < utils.max_ensemble_size:
                less_than_max = True
            if less_than_max:
                print('Warning: Ensemble created with fewer than',
                      utils.max_ensemble_size, 'trees.')
            print('---------------------------')

        # Save experimental data to file...
        print('Saving data to file...')
        savedir_ = utils.savedir + '\seed-' + str(seed)
        try:
            os.makedirs(savedir_)
        except FileExistsError:
            pass

        # ...for baseline ensembles
        test_errors_ = np.array(test_errors_)
        np.savetxt(savedir_ + '\\test-errors-baseline.csv', test_errors_,
                   fmt='%.18f', delimiter=',')

        # ...for poisoned ensembles
        test_errors_p_ = np.array(test_errors_p_)
        np.savetxt(savedir_ + '\\test-errors-poisoned.csv', test_errors_p_,
                   fmt='%.18f', delimiter=',')

        # Plot average errors and confidence intervals...
        utils.plot_statistical_significance(test_errors_, test_errors_p_, seed)
        print('Done.')
        print('---------------------------')

    s = time.time() - start
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    print('Runtime: {0:d} hours, {1:d} minutes, {2:.2f} seconds.'
          .format(int(h), int(m), s))
    plt.show()
    print('Done.')

if __name__ == '__main__':
    main()

