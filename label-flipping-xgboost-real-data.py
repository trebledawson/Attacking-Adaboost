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
# Experiments are performed on a variety of real-world datasets. A 100-fold     #
# out-of-bootstrap validation is performed on each dataset. For each fold, the  #
# testing dataset is uniformly drawn from the total dataset. Over each fold, an #
# Adaboost classifier built on decision trees is trained, first on the          #
# baseline, unpoisoned dataset, and then on the dataset poisoned by             #
# label-flipping. The average error over n_runs is reported, along with a 95%   #
# confidence interval.                                                          #
# ############################################################################# #

import os
import time
from datetime import date
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils


# Experimental parameters
dataset = 'digits'
classifier = 'tree'
depth = 5
n_folds = 100
if n_folds >= 30:
    test_type = 'Z'
else:
    test_type = 'T'

# Save results in this local directory
savedir = '.\Results\\' + str(date.today()) + '\\real-world-data\\xgboost\\' + \
          dataset
if utils.flip_original:
    savedir += '\\flip-original-'
else:
    savedir += '\\add-copies-'
savedir += str(int(100 * utils.percent_poison)) + '-percent'

try:
    os.makedirs(savedir)
except FileExistsError:
    path_exists = True
    slug = 1
    while path_exists:
        try:
            os.makedirs(savedir + '-' + str(slug))
            path_exists = False
        except FileExistsError:
            slug += 1
    savedir += '-' + str(slug)

def main():
    start = time.time()
    test_errors_ = []
    test_errors_p_ = []
    print('Loading data...')
    data, labels = utils.load_dataset(dataset)

    for fold in range(n_folds):
        print('Dataset:', dataset, '| Fold:', fold + 1)

        print('Splitting data...')
        d_train, d_test, y_train, y_test = tts(data, labels,
                                               test_size=0.3,
                                               shuffle=True,
                                               stratify=None)

        print('Generating poisoned dataset...')
        flip_idx = np.random.choice(range(len(y_train)),
                                    size=int(utils.percent_poison
                                             * len(y_train)),
                                    replace=False)

        if utils.flip_original:
            d_train_p = d_train
            y_train_p = deepcopy(y_train)
            for idx in flip_idx:
                y_train_p[idx] = np.random.choice([y for y in np.unique(y_train)
                                                   if y != y_train[idx]])
        else:
            # N.B. Unsuitable for non-binary problems.
            d_p = [d_train[idx, :] for idx in flip_idx]
            y_p = [1 if y_train[idx] == 0 else 0 for idx in flip_idx]
            d_train_p = np.vstack((d_train, d_p))
            y_train_p = np.hstack((y_train, y_p))

        print('Training size:', len(y_train))
        print('Poisoned size:', len(y_train_p))
        print('---Percent poisoned :', 100 * utils.percent_poison)
        print('Testing size:', len(y_test))

        # Train and test on baseline data
        print('Fitting model on training data...')
        xgb_ensemble = utils.make_ensemble(boost='xgboost')
        xgb_ensemble.fit(d_train, y_train, verbose=False)

        print('Predicting on test data...')
        test_errors = []
        for n in range(1, utils.max_ensemble_size):
            pred = xgb_ensemble.predict(d_test, ntree_limit=n)
            test_errors.append(1. - accuracy_score(y_test, pred))

        # Train and test on poisoned data
        print('Fitting model on poisoned training data...')
        xgb_ensemble_p = utils.make_ensemble(boost='xgboost')
        xgb_ensemble_p.fit(d_train_p, y_train_p, verbose=False)

        print('Predicting on test data...')
        test_errors_p = []
        for n in range(1, utils.max_ensemble_size):
            pred = xgb_ensemble_p.predict(d_test, ntree_limit=n)
            test_errors_p.append(1. - accuracy_score(y_test, pred))

        # Store individual run errors for statistical analysis
        test_errors_.append(test_errors)
        test_errors_p_.append(test_errors_p)

        print('Baseline error:', test_errors[-1],
              '| # Trees:', utils.max_ensemble_size)
        print('Poisoned error:', test_errors_p[-1],
              '| # Trees:', utils.max_ensemble_size)

        # Plot individual run errors
        if utils.plot_runs:
            print('Plotting...')
            plt.figure()
            plt.title('Testing on XGBoost')
            plt.plot(range(1, len(test_errors) + 1),
                     test_errors,
                     label='Baseline')
            plt.plot(range(1, len(test_errors_p) + 1),
                     test_errors_p,
                     label='Poisoned')
            plt.ylabel('Test Error')
            plt.grid()
            plt.legend()
            plt.show()
        print('---------------------------')

    # Save experimental data to file...
    print('Saving data to file...')

    # ...for baseline ensembles
    test_errors_ = np.array(test_errors_)
    np.savetxt(savedir + '\\test-errors-baseline.csv', test_errors_,
               fmt='%.18f', delimiter=',')

    # ...for poisoned ensembles
    test_errors_p_ = np.array(test_errors_p_)
    np.savetxt(savedir + '\\test-errors-poisoned.csv', test_errors_p_,
               fmt='%.18f', delimiter=',')

    # Plot average errors and confidence intervals...
    utils.plot_statistical_significance_real(test_errors_, test_errors_p_,
                                             dataset, n_folds, savedir,
                                             type=test_type)
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

