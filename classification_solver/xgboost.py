import os
import pickle
import random

import numpy as np
import optuna
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                   cross_val_score, cross_validate)
from tqdm import tqdm
from xgboost import XGBClassifier

from .base import ClassificationModel


class XGBoostModel(ClassificationModel):
    def __init__(
        self,
        name: str = "XGBoost",
        task_name: str = "",
        n_classes: int = 2,
        seed: int = 2024,
        model_dir=None,
        n_dim=None,
        results_dir=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        if task_name == "anomaly_classification":
            self.max_depth = 13
            self.n_estimators = 293
            self.learning_rate = 0.8021654216265983
            self.reg_alpha = 0.11282621272072024
            self.reg_lambda = 2.4680054712435116
            self.gamma = 0.11540930574666763
        elif task_name == "k_classification":
            self.max_depth = 10
            self.n_estimators = 152
            self.learning_rate = 0.02553040552398947
            self.reg_alpha = 1.96062313615598
            self.reg_lambda = 6.369816415133323
            self.gamma = 2.649574766332463
        else:   
            # 默认参数设置
            self.max_depth = 6
            self.n_estimators = 100
            self.learning_rate = 0.3
            self.reg_alpha = 0
            self.reg_lambda = 1
            self.gamma = 0
        
        self.random_state = seed
        self.tree_method = "hist"
        
        # 初始化模型
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            random_state=self.random_state,
            tree_method=self.tree_method,
            n_jobs=-1,
        )
        
        # 参数优化范围
        self.param_range = {
            "max_depth": (3, 15),
            "n_estimators": (50, 500),
            "learning_rate": (0.01, 1),
            "reg_alpha": (0, 10),
            "reg_lambda": (0, 10),
            "gamma": (0, 10),
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

            self.model = XGBClassifier(
                random_state=self.random_state,
                tree_method=self.tree_method,
                **best_params
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
                if key in ["max_depth", "n_estimators"]:
                    trial_args_dict[key] = trial.suggest_int(key, min_val, max_val)
                else:
                    trial_args_dict[key] = trial.suggest_float(key, min_val, max_val)

            model = XGBClassifier(
                random_state=self.random_state,
                tree_method=self.tree_method,
                **trial_args_dict
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
        study.optimize(objective, n_trials=50, n_jobs=-1)
        return study.best_params

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

    def save(self, save_path: str) -> None:
        assert self.is_trained
        with open(os.path.join(save_path, "XGBoost_classifier.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_path: str) -> None:
        model_file = os.path.join(model_path, "XGBoost_classifier.pkl")
        assert os.path.exists(model_file)
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
