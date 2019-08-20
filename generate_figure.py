import sys
import pandas as pd
import matplotlib.pyplot as plt
import utils

def main(dir, seed):
    data = pd.read_csv(dir + '\\test-errors-baseline.csv')
    data_p = pd.read_csv(dir + '\\test-errors-poisoned.csv')

    utils.plot_statistical_significance(data, data_p, seed, boost='XGBoost')

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])