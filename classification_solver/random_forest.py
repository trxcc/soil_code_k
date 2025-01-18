import os
import pickle
import random

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                   cross_val_score, cross_validate)

from .base import ClassificationModel


class RandomForestModel(ClassificationModel):
    def __init__(
        self,
        name: str = "RandomForest",
        task_name: str = "default",
        seed: int = 2024,
        n_classes: int = 2,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        if task_name == "anomaly_classification":
            self.max_depth = 11
            self.n_estimators = 10
            self.min_samples_split = 7
        elif task_name == "k_classification":
            self.max_depth = 11
            self.n_estimators = 64
            self.min_samples_split = 2
        else:
            # 为不同任务设置默认参数
            self.max_depth = 10
            self.n_estimators = 100
            self.min_samples_split = 2
            # 可以根据需要添加其他任务的默认参数
        
        self.random_state = seed
        self.model = RandomForestClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.param_range = {
            "max_depth": (1, 20),
            "n_estimators": (10, 300),
            "min_samples_split": (2, 10),
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
                X_train=X_train,
                y_train=y_train,
                model_args=self.param_range,
                optimize_method=optimize_method,
            )

            if self.results_dir is not None:
                with open(
                    os.path.join(self.results_dir, "best_params.pkl"), "wb+"
                ) as f:
                    pickle.dump(file=f, obj=best_params)

            self.model = RandomForestClassifier(
                random_state=self.random_state, **best_params
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
        def objective(trial):
            trial_args_dict = dict()
            for key, (min_val, max_val) in model_args.items():
                trial_args_dict.update({key: trial.suggest_int(key, min_val, max_val)})

            model = RandomForestClassifier(
                random_state=self.random_state, **trial_args_dict
            )
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
            return final_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, n_jobs=-1)

        return study.best_trial.params

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_args: dict,
        optimize_method: str = "bayesopt",
    ) -> dict:
        if optimize_method.lower() == "bayesopt":
            return self._optimize_hyperparameters_optuna(X_train, y_train, model_args)
        elif optimize_method.lower() == "gridsearch":
            return self._optimize_hyperparameters_grid_search(
                X_train, y_train, model_args
            )
        else:
            raise NotImplementedError("未知的优化方法。")

    def _optimize_hyperparameters_grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:
        param_ranges = {
            "max_depth": (1, 20, 10),
            "n_estimators": (10, 300, 30),
            "min_samples_split": (2, 10, 5),
        }

        grid = {}
        for param, (min_val, max_val, num) in param_ranges.items():
            values = np.linspace(min_val, max_val, num, dtype=int)
            grid[param] = values.tolist()

        param_grid = list(ParameterGrid(grid))
        if len(param_grid) > 100:
            param_grid = random.sample(param_grid, 100)

        model = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=[{k: [v] for k, v in p.items()} for p in param_grid],
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)
        print("最佳参数:", grid_search.best_params_)
        print("最佳准确率:", grid_search.best_score_)

        return grid_search.best_params_

    def save(self, save_path: str) -> None:
        assert self.is_trained
        with open(os.path.join(save_path, "RandomForestClassifier.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_path: str) -> None:
        model_file = os.path.join(model_path, "RandomForestClassifier.pkl")
        assert os.path.exists(model_file)
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
