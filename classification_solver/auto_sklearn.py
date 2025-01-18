import os
import pickle

import numpy as np
from autosklearn.classification import AutoSklearnClassifier

from .base import ClassificationModel


class AutoSklearnModel(ClassificationModel):
    def __init__(
        self,
        name: str = "AutoSklearnClassifier",
        task_name: str = "",
        n_classes: int = 2,
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        # 使用 AutoSklearnClassifier 替代回归版本的 AutoSklearnRegressor
        self.model = AutoSklearnClassifier(memory_limit=307200, seed=seed)
        self.seed = seed
        self.model_dir = model_dir
        self.results_dir = results_dir

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_hyperparams: bool = True,
        optimize_method: str = "BayesOpt",
    ) -> None:
        super().fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)

    def save(self, save_path: str) -> None:
        assert self.is_trained
        with open(
            os.path.join(save_path, f"AutoSklearnClassifier-{self.seed}.pkl"), "wb"
        ) as f:
            pickle.dump(self.model, f)

    def load(self, model_dir: str) -> None:
        model_path = os.path.join(model_dir, f"AutoSklearnClassifier-{self.seed}.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
