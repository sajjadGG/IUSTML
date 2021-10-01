import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class NaiveBayes(Model):
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, likelihood="gauusian", priors=None, **kwargs):
        """fit Naive bayes model with given data and specified parameters

        Args:
            X (np.array): training data
            y (np.array): training label
            likelihood (str, optional): distribution family of likelihood function. Defaults to 'gauusian'. supported types
            gauusian, multinomial, custom in case of custom a likelihood_fit function should be passed as keyword argument
            priors (np.array, optional): specify the prior distribution of classes. it should be the same size of number of
            classes and it should sum to one. Defaults to None.
            alpha (float, kwargs): smoothing used in multinomial NB
        """

        if likelihood == "multinomial":
            pass

    def score(self, X, y):
        """compute MSE score by default

        Args:
            X (np.array): features
            y (np.array): labels
        """
        pass

    def predict(self, X):
        """predict using fitted model and it should be called after fit

        Args:
            X (np.array): features

        Returns:
            [np.array]: predicted classes
        """

        assert self.is_fitted

        pass
