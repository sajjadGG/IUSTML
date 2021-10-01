import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def encode_onehot(y):
    """return a dense matrix of y encoded in onehot scheme

    Args:
        y (np.array): vector to be encoded
    """

    classes = np.unique(y)
    class_mapping = {k: v for v, k in enumerate(classes)}
    Y = np.zeros((y.shape[0], len(classes)))
    for i in range(y.shape[0]):
        Y[i][class_mapping[y[i]]] = 1
    return Y, classes, class_mapping


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
        self.classes = None  # from N to classes
        self.feature_likelihood = None
        self.number_of_features = None
        self.class_mapping = {}  # from give classes to N

    def __initialize(self, X, y):
        Y, self.classes, self.class_mapping = encode_onehot(y)
        return Y

    def fit(self, X, y, likelihood="gauusian", priors=None, **kwargs):
        """fit Naive bayes model with given data and specified parameters

        Args:
            X (np.array): training data in the format of number_of_examples X number_of_features
            y (np.array): training label
            likelihood (str, optional): distribution family of likelihood function. Defaults to 'gauusian'. supported types
            gauusian, multinomial, custom in case of custom a likelihood_fit function should be passed as keyword argument
            priors (np.array, optional): specify the prior distribution of classes. it should be the same size of number of
            classes and it should sum to one. Defaults to None.
            alpha (float, kwargs): smoothing used in multinomial NB
        """
        Y = self.__initialize(X, y)
        self.number_of_features = X.shape[1]

        if likelihood == "multinomial":
            self.alpha = kwargs.get("alpha", 0.000000001)
            assert self.alpha > 0
            # self.feature_likelihood = np.zeros(
            #     (len(self.classes), self.number_of_features)
            # )
            total_class_count = np.zeros(len(self.classes))
            self.feature_likelihood = Y.T @ X

            # NOTE: summing over feature value hence values must be indicative of importance not perpendicular values like categories in that case first make it one-hot

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
