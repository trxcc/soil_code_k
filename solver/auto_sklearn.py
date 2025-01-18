import os
import pickle

import numpy as np
from autosklearn.regression import AutoSklearnRegressor

import wandb

from .base import RegressionModel


class AutoSklearnModel(RegressionModel):
    def __init__(
        self,
        name: str = "AutoSklearn",
        task_name: str = "",
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model = AutoSklearnRegressor(memory_limit=307200, seed=seed)
        self.seed = seed
        self.model_dir = model_dir
        self.results_dir = results_dir

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_hyperparams: bool = False,
        optimize_method=None,
    ) -> None:
        super().fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def save(self, save_path) -> None:
        assert self.is_trained
        with open(
            os.path.join(save_path, f"AutoSklearnModel-{self.seed}.pkl"), "wb"
        ) as f:
            pickle.dump(self.model, f)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, f"AutoSklearnModel-{self.seed}.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
