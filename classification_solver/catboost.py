import os
import pickle
import random

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, average_precision_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score, cross_validate)

from .base import ClassificationModel

record_i = 1


class CatBoostModel(ClassificationModel):
    def __init__(
        self,
        name: str = "CatBoost",
        task_name: str = "",
        n_classes: int = 2,
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir

        if task_name == "anomaly_classification":
            self.max_depth = 8
            self.n_estimators = 226
            self.learning_rate = 0.22161273261465986
        elif task_name == "k_classification":
            self.max_depth = 6
            self.n_estimators = 240
            self.learning_rate = 0.33261971025506376
        else:
            # 默认参数设置
            self.max_depth = 6
            self.n_estimators = 100
            self.learning_rate = 0.1

        self.task_type = "CPU"
        self.devices = "0"
        self.random_state = seed
        self.model = CatBoostClassifier(
            max_depth=self.max_depth,
            thread_count=64,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            task_type=self.task_type,
            devices=self.devices,
        )
        self.param_range = {
            "max_depth": (1, 8),
            "n_estimators": (10, 300),
            "learning_rate": (0, 1),
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_hyperparams: bool = True,
        optimize_method: str = "BayesOpt",
    ) -> None:
        super().fit(X_train, y_train)
        if optimize_hyperparams:
            best_params = self.optimize_hyperparameters(
                X_train=X_train, y_train=y_train, model_args=self.param_range
            )

            if self.results_dir is not None:
                with open(
                    os.path.join(self.results_dir, "best_params.pkl"), "wb+"
                ) as f:
                    pickle.dump(file=f, obj=best_params)

            self.model = CatBoostClassifier(
                random_state=self.random_state,
                task_type=self.task_type,
                devices=self.devices,
                **best_params,
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)

    def _optimize_hyperparameters_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:
        def object_func(trial):
            trial_args_dict = dict()
            for key, (min_val, max_val) in model_args.items():
                if key == "learning_rate":
                    trial_args_dict[key] = trial.suggest_float(key, min_val, max_val)
                else:
                    trial_args_dict[key] = trial.suggest_int(key, min_val, max_val)

            model = CatBoostClassifier(
                random_state=self.random_state,
                thread_count=64,
                task_type=self.task_type,
                devices=self.devices,
                **trial_args_dict,
            )
            
            # 使用 cross_validate 替代 cross_val_score
            cv_results = cross_validate(
                model, 
                X_train, 
                y_train, 
                cv=5,
                scoring={
                    'accuracy': 'accuracy',
                    'roc_auc': 'roc_auc',
                    'average_precision': 'average_precision'
                }
            )
            
            # 计算每个指标的平均值
            final_score = (
                cv_results['test_accuracy'].mean() * 0.4 +
                cv_results['test_roc_auc'].mean() * 0.3 +
                cv_results['test_average_precision'].mean() * 0.3
            )
            
            global record_i
            print(f"Trial {record_i}: {final_score}")
            record_i += 1
            return final_score

        study = optuna.create_study(direction="maximize")
        study.optimize(object_func, n_trials=100, n_jobs=1)
        return study.best_trial.params

    def optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:
        return self._optimize_hyperparameters_optuna(X_train, y_train, model_args)

    def save(self, save_path: str) -> None:
        assert self.is_trained
        with open(os.path.join(save_path, "CatBoost_classifier.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_path: str) -> None:
        model_file = os.path.join(model_path, "CatBoost_classifier.pkl")
        assert os.path.exists(model_file)
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
