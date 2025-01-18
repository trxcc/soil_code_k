import os
import pickle
import random

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score)

import wandb

from .base import RegressionModel


class RandomForestModel(RegressionModel):
    def __init__(
        self,
        name: str = "RandomForest",
        task_name: str = "MurA",
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir
        if task_name == "MurA":
            self.max_depth = 18
            self.n_estimators = 123
            self.min_samples_split = 2
        elif task_name == "GlcN":
            self.max_depth = 18
            self.n_estimators = 270
            self.min_samples_split = 2
        elif task_name == "A_GN":
            self.max_depth = 18
            self.n_estimators = 277
            self.min_samples_split = 3
        elif task_name == "F_GN":
            self.max_depth = 20
            self.n_estimators = 219
            self.min_samples_split = 4
        elif task_name == "k":
            self.max_depth = 9
            self.n_estimators = 104
            self.min_samples_split = 2
        elif task_name == "k-240906":
            self.max_depth = 12
            self.n_estimators = 54
            self.min_samples_split = 2
        elif task_name == "k-241004":
            self.max_depth = 20
            self.n_estimators = 35
            self.min_samples_split = 2
        elif task_name == "em-241121":
            self.max_depth = 10
            self.n_estimators = 132
            self.min_samples_split = 3
        elif task_name == "fPOC":
            self.max_depth = 13
            self.n_estimators = 25
            self.min_samples_split = 4
        elif task_name == "MAOC_N":
            self.max_depth = 8
            self.n_estimators = 263
            self.min_samples_split = 3
        elif task_name == "POC_N":
            self.max_depth = 13
            self.n_estimators = 278
            self.min_samples_split = 3
        else:
            self.max_depth = 18
            self.n_estimators = 123
            self.min_samples_split = 2
        self.random_state = seed
        self.model = RandomForestRegressor(
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

            self.model = RandomForestRegressor(
                random_state=self.random_state, **best_params
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def _optimize_hyperparameters_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:

        def object_func(trial):
            trial_args_dict = dict()
            # print(model_args.items())
            for key, (min_val, max_val) in model_args.items():
                trial_args_dict.update({key: trial.suggest_int(key, min_val, max_val)})

            model = RandomForestRegressor(
                random_state=self.random_state, **trial_args_dict
            )
            score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
            # wandb.log({'trial_score': score})
            return score

        study = optuna.create_study(direction="maximize")
        trial_num = 500
        study.optimize(object_func, n_trials=trial_num)

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
            raise NotImplementedError("Unknown optimizing method.")

    def _optimize_hyperparameters_grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:

        def create_param_grid(param_ranges, num_combinations=100):
            """生成指定数量的参数组合。

            参数:
            param_ranges (dict): 参数的名称和它们的 (min, max, num) 范围的字典。
            num_combinations (int): 期望的参数组合数。

            返回:
            list of dict: 参数名和生成的参数组合的列表。
            """
            grid = {}
            total_combinations = 1
            for param, (min_val, max_val, num) in param_ranges.items():
                # if isinstance(min_val, int) and isinstance(max_val, int):
                # 产生整数序列
                values = np.linspace(min_val, max_val, num, dtype=int).tolist()
                # else:
                #     # 产生浮点数序列
                #     values = np.linspace(min_val, max_val, num).tolist()
                grid[param] = values
                total_combinations *= len(values)

            # 生成所有可能的组合
            param_grid = list(ParameterGrid(grid))

            # 如果总组合数超过 num_combinations，随机选择
            if total_combinations > num_combinations:
                param_grid = random.sample(param_grid, num_combinations)

            # 确保每个参数的值都是列表
            for param_dict in param_grid:
                for key in param_dict:
                    param_dict[key] = [param_dict[key]]

            return param_grid

        param_ranges = {
            "max_depth": (1, 20, 10),
            "n_estimators": (10, 300, 100),
            "min_samples_split": (2, 10, 8),
        }

        param_grid = create_param_grid(param_ranges, num_combinations=100)
        print(param_grid)
        # assert 0, param_grid

        model = RandomForestRegressor(random_state=self.random_state)

        # 定义评分函数
        scorer = make_scorer(r2_score)

        # 创建 GridSearchCV 对象
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            cv=5,
            n_jobs=-1,  # 使用所有可用的CPU核心
            verbose=2,
        )  # 显示搜索过程中的详细信息

        # 执行网格搜索
        grid_search.fit(X_train, y_train)

        # 输出最优参数和最优分数
        print("Best parameters:", grid_search.best_params_)
        print("Best R2 score:", grid_search.best_score_)

        return grid_search.best_params_

    def save(self, save_path) -> None:
        assert self.is_trained
        with open(os.path.join(save_path, "RandomForestModel.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "RandomForestModel.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
