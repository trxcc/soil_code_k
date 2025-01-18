from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class RegressionModel(ABC):
    """A base solver for regression problem

    Returns a regression model.

    :param name: the name of the regression model.
    :param X_train: the features of training data.
    :param y_train: the labels of training data.
    :param is_trained: a flag to indicate that if the model has been trained.
    """

    def __init__(self, name: str) -> None:

        self.name = name
        self.X_train = None
        self.y_train = None
        self.is_trained = False  # A flag to indicate that if the model has been trained

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_hyperparams: bool = False,
    ) -> None:
        """
        Train the model.

        :param X_train (array-like): the features of training data.
        :param y_train (array-like): the labels of training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained model to predict.

        Returns the prediction given X_test.

        :params X_test (array-like): the features of test data.
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the performance of the model.

        Returns R2 score corresponding to X_test and y_test.


        :param X_test (array-like): the features of test data.
        :param y_test (array-like): the labels of test data.
        """
        # assert self.is_trained and self.X_train is not None and self.y_train is not None
        y_pred = self.predict(X_test)
        # return np.sqrt(mean_squared_error(y_test, y_pred))
        return r2_score(y_test, y_pred)

    def optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:
        """Optimize the hyper-parameters of the model.

        Returns arguments of the model that has been optimized.

        :param X_train (array-like): the features of training data.
        :param y_train (array-like): the labels of training data.
        :param model_args: a dict that contains arguments to be optimized.
        """
        pass

    @abstractmethod
    def save(self, save_path) -> None:
        """Save the model.

        :param save_path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, model_path):
        pass
