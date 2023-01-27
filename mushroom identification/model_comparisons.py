"""Author: Jack King

Runs and prints all combinations of given categorical encoders and scikit machine learning classifier models on a dataset.

This code specifically encodes in Binary, Ordinal, and OneHot and runs KNeighborns, DecisionTree, and LogisitcRegression on the mushroom dataset"""


import numpy as np
import pandas as pd
import math
import re
import time
import functools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class BinaryEncoder():
    """Takes a dataframe of values
        Binary encodes values"""

    def __init__(self):
        """initializes the global variables"""
        # Ordinal encoder for all values
        self.ordinalEncoder = OrdinalEncoder()
        # max bits for each column in the dataset
        self.maxBits = {}

    def transform(self, df):
        """using the ordinal encoder and maxbits already created in
        fit_transform transforms the df into binary

            df: dataframe of same dataset as fittied dataframe

            returns: the binary encoded dataframe"""

        ordinalData = pd.DataFrame(self.ordinalEncoder.transform(df))
        new_data = pd.DataFrame()
        for column in ordinalData.columns:
            maxBits = self.maxBits[column]
            for bit in range(maxBits):
                new_data[str(column) + str(bit)] = ordinalData[column] % 2
                ordinalData[column] //= 2
        return new_data


    def fit_transform(self, df):
        """fit an ordinal encoder and maxBits dictionary
        to a dataset and binary encodes the dataframe

            df: dataframe to fit on and transform

            returns: the binary encoded dataframe"""

        ordinalData = pd.DataFrame(self.ordinalEncoder.fit_transform(df))
        new_data = pd.DataFrame()
        for column in ordinalData.columns:
            max = ordinalData[column].max()
            bits = int(math.log(max+1,2))
            self.maxBits[column] = bits
            for bit in range(bits):
                new_data[str(column) + str(bit)] = ordinalData[column] % 2
                ordinalData[column] //= 2
        return new_data

class ComparisonDoer():
    """takes a dataset and compares the effectiveness of combinations of encoders and models"""

    def __init__(self, dfName, columnNames, targetName, encoders, models):
        """starts prepping process and initializes global variables"""
        mainDf = pd.read_csv(dfName, names = columnNames)
        self.encoders = encoders
        self.models = models
        self.prep(mainDf, targetName)

    def test_train_split(self, mydf, tratio, target):
        """splits the dataset into test and train randomly and returns both"""

        splitter = StratifiedShuffleSplit(test_size=tratio, random_state=42)
        train_index, test_index = next(splitter.split(mydf, mydf[target]))
        strat_train = mydf.iloc[train_index]
        strat_test = mydf.iloc[test_index]
        return strat_train, strat_test

    def prep(self, mainDf, targetName):
        """ splits test and train into inputs and outputs
            targetName: the name of the target column"""

        train, test = self.test_train_split(mainDf, .2, targetName)
        self.train_targets = train[targetName].apply(self.toTargets)
        self.test_targets = test[targetName].apply(self.toTargets)
        self.train_inputs = train.drop(targetName,axis=1)
        self.test_inputs = test.drop(targetName,axis=1)

    def print_conf_matrix(self, targets, outputs):
        """prints a confusion matrix for a predicted Classifier
            targets: targets from original DataFrame
            outputs: the predicted results from the classifier"""

        cm = confusion_matrix(targets, outputs)
        print("Confusion Matrix:")
        print("     PN PP")
        print("AN: "+ str(cm[0]))
        print("AP: "+ str(cm[1]))

    def toTargets(self, val):
        """turns the targets into binary"""
        if val == 'e':
            return 1
        else:
            return 0


    def timer(func):
        """Print the length it takes to run function func"""
        @functools.wraps(func)
        def wrapper_timer(*args):
            start = time.perf_counter()
            value = func(*args)
            end = time.perf_counter()
            duration = end - start
            print(f"Finished {func.__name__!r} in {duration:.4f} secs")
            return value
        return wrapper_timer

    @timer
    def predict(self, train_inputs, test_inputs, model):
        """fits a model to the train and then predicts values based on the test
            model: machine learning classifier
            train_inputs: inputs for training the model
            test_inputs: inputs for predicting values to test"""
        model.fit(train_inputs, self.train_targets)
        outputs = model.predict(test_inputs)
        print("Mean test accuracy:", model.score(test_inputs, self.test_targets))
        self.print_conf_matrix(self.test_targets, outputs)

    def main(self):
        """for each encoder, runs each model and prints confusion matrix and accuracy"""
        for enc in self.encoders:
            print(enc.__class__.__name__)
            train_inputs, test_inputs = enc.fit_transform(self.train_inputs), enc.transform(self.test_inputs)
            for model in self.models:
                print(model.__class__.__name__)
                self.predict(train_inputs, test_inputs, model)
            print('\n')


names = ['Class','cap-shape', 'cap-surface','cap-color','bruises','odor','gill-attachment',
'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-abv-ring',
'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
'veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
'habitat']
encoders = [OneHotEncoder(sparse = False), OrdinalEncoder(), BinaryEncoder()]
# Trained models in november code
models = [LogisticRegression(penalty='none', solver='saga', random_state=42),
DecisionTreeClassifier(max_depth = 15, min_samples_leaf = 2), KNeighborsClassifier(n_neighbors = 1)]

comparison = ComparisonDoer("mushrooms.csv", names, 'Class', encoders, models)
comparison.main()
