import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from utils import SklearnAttack

def main():
    print('Generating data...')
    data, labels = make_classification(n_samples=1000,
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
                                       random_state=42)

    print('Splitting data...')
    d_train, d_test, y_train, y_test = tts(data, labels,
                                           test_size=0.3,
                                           random_state=42,
                                           shuffle=True,
                                           stratify=None)

    print('Initializing ensemble base classifier...')
    BaseClassifier = make_classifier()

    print('Poisoning training data...')
    budget = 0.1
    n_points = int(budget * len(y_train))
    mesh = 5
    bound = mesh + 10
    boundary = np.array([[-bound, bound],
                         [bound, -bound]])
    attack = SklearnAttack(BaseClassifier, opt_method='Powell', step_size=5,
                           max_steps=1, boundary=boundary)
    attack.fit(d_train, y_train)
    x_attack, y_attack = attack.attack(n_points)
    d_train_p = np.vstack((d_train, x_attack))
    y_train_p = np.concatenate((y_train, y_attack))

    print('Testing poisoning attack on decision tree classifier...')
    tree = make_classifier()
    tree.fit(d_train, y_train)
    y_predict_t = tree.predict(d_test)
    score = accuracy_score(y_predict_t, y_test)

    tree_p = make_classifier()
    tree_p.fit(d_train_p, y_train_p)
    y_predict_t_p = tree_p.predict(d_test)
    score_p = accuracy_score(y_predict_t_p, y_test)

    plt.figure()
    plt.title('Accuracy of single decision tree')
    plt.bar([0, 1], [score, score_p])
    plt.xticks([0, 1], ('Normal', 'Poisoned'))

    ensemble = make_ensemble(BaseClassifier)

    print('Fitting model on training data...')
    ensemble.fit(d_train, y_train)

    test_errors = []

    print('Predicting on test data...')
    for prediction in ensemble.staged_predict(d_test):
        test_errors.append(1. - accuracy_score(y_test, prediction))

    n_trees = len(ensemble)
    ensemble_errors = ensemble.estimator_errors_[:n_trees]

    print('Plotting...')
    plt.figure()
    plt.suptitle('Testing on AdaBoost')
    plt.subplot(311)
    plt.plot(range(1, n_trees + 1),
             test_errors,
             label='Normal')
    plt.legend()
    plt.grid()
    plt.ylabel('Test Error')

    plt.subplot(312)
    plt.plot(range(1, n_trees + 1),
             ensemble_errors,
             c='black', label='Normal')
    plt.legend()
    plt.grid()
    plt.ylabel('Classifier Error')

    ensemble_p = make_ensemble(BaseClassifier)

    print('Fitting model on poisoned training data...')
    ensemble_p.fit(d_train_p, y_train_p)

    test_errors_p = []
    print('Predicting on test data...')
    for prediction in ensemble_p.staged_predict(d_test):
        test_errors_p.append(1. - accuracy_score(y_test, prediction))

    n_trees_p = len(ensemble_p)
    ensemble_errors_p = ensemble_p.estimator_errors_[:n_trees]

    print('Plotting...')
    plt.subplot(311)
    plt.plot(range(1, n_trees_p + 1),
             test_errors_p,
             label='Poisoned')
    plt.legend()
    plt.subplot(313)
    plt.plot(range(1, n_trees_p + 1),
             ensemble_errors_p,
             c='black', label='Poisoned')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of Trees')
    plt.ylabel('Classifier Error')

    print('Normal accuracy:', test_errors[-1])
    print('Poisoned accuracy:', test_errors_p[-1])
    plt.show()
    print('Done.')

def make_classifier():
    tree = DecisionTreeClassifier(criterion='gini',
                                  splitter='best',
                                  max_depth=5,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None,
                                  random_state=7,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  class_weight=None,
                                  presort=False)

    return tree

def make_ensemble(BaseClassifier):
    print('Initializing boosted classifier...')
    ensemble = AdaBoostClassifier(BaseClassifier,
                                  n_estimators=1000,
                                  learning_rate=1.0,
                                  algorithm='SAMME.R',
                                  random_state=3901)

    return ensemble

if __name__ == '__main__':
    main()