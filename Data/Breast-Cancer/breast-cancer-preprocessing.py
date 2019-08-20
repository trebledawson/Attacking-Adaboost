# #################################################################### #
# breast-cancer-preprocessing.py                                       #
# Author: Glenn Dawson                                                 #
# -------------------------------                                      #
# This file preprocesses the UCI Breast Cancer dataset.                #
# #################################################################### #

import pandas as pd

def main():
    names = ['class',
             'age',
             'menopause',
             'tumor_size',
             'inv_nodes',
             'node_caps',
             'deg_malig',
             'breast',
             'breast_quad',
             'irradiat']
           
    data = pd.read_csv('breast-cancer.data', header=None, names=names)
    
    # Class
    data.replace(['no-recurrence-events', 'recurrence-events'], [0, 1], inplace=True)
    
    # Age
    data.replace(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
                 
    # Menopause
    data.replace(['lt40', 'ge40', 'premeno'], [1, 2, 3], inplace=True)
    
    # Tumor size
    data.replace(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
    
    # Inv-nodes
    data.replace(['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], inplace=True)
    
    # Node caps / Irradiat
    data.replace(['yes', 'no'], [0, 1], inplace=True)
    
    # Breast
    data.replace(['left', 'right'], [0, 1], inplace=True)
    
    # Breast quad
    data.replace(['left_up', 'left_low', 'right_up', 'right_low', 'central'],
                 [1, 2, 3, 4, 5], inplace=True)
    
    # Remove instances with missing values
    data = data[data.node_caps != '?']
    data = data[data.breast_quad != '?']
    
    # Save
    data.to_csv('breast-cancer-numeric.csv')
    
if __name__ == '__main__':
    main()
    
