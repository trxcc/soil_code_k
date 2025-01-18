import os

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
import pickle
import random

import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score)

import wandb

from .base import RegressionModel

record_i = 1


class CatBoostModel(RegressionModel):
    def __init__(
        self,
        name: str = "CatBoost",
        task_name: str = "",
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir
        if task_name == "MurA":
            self.max_depth = 6
            self.n_estimators = 157
            self.learning_rate = 0.20707969404668414
        elif task_name == "GlcN":
            self.max_depth = 5
            self.n_estimators = 294
            self.learning_rate = 0.09152523997275597
        elif task_name == "A_GN":
            self.max_depth = 5
            self.n_estimators = 219
            self.learning_rate = 0.2374548978998544
        elif task_name == "F_GN":
            self.max_depth = 8
            self.n_estimators = 198
            self.learning_rate = 0.2413167436413624
        elif task_name == "k":
            self.max_depth = 7
            self.n_estimators = 10
            self.learning_rate = 0.9584039205677916
        elif task_name == "k-241004":
            self.max_depth = 8
            self.n_estimators = 259
            self.learning_rate = 0.070761287849903
        elif task_name == "em-241121":
            self.max_depth = 7
            self.n_estimators = 231
            self.learning_rate = 0.30485885551382985
        elif task_name == "fPOC-all_data":
            self.max_depth = 3
            self.n_estimators = 45 
            self.learning_rate = 0.0969660471647072
        elif task_name == "MAOC_N-all_data":
            self.max_depth = 3
            self.n_estimators = 103 
            self.learning_rate = 0.2915143290359795
        elif task_name == "POC_N-all_data":
            self.max_depth = 4
            self.n_estimators = 263 
            self.learning_rate = 0.433933196123283
        else:
            self.max_depth = 5
            self.n_estimators = 294
            self.learning_rate = 0.09152523997275597

        self.task_type = "CPU"
        self.devices = "0"
        self.random_state = seed
        self.model = CatBoostRegressor(
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

            self.model = CatBoostRegressor(
                random_state=self.random_state,
                task_type=self.task_type,
                devices=self.devices,
                **best_params,
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test).reshape(-1, 1)

    def _optimize_hyperparameters_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:

        def object_func(trial):
            trial_args_dict = dict()
            # print(model_args.items())
            for key, (min_val, max_val) in model_args.items():
                if key == "learning_rate":
                    trial_args_dict.update(
                        {key: trial.suggest_float(key, min_val, max_val)}
                    )
                else:
                    trial_args_dict.update(
                        {key: trial.suggest_int(key, min_val, max_val)}
                    )

            model = CatBoostRegressor(
                random_state=self.random_state,
                thread_count=64,
                task_type=self.task_type,
                devices=self.devices,
                **trial_args_dict,
            )
            score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
            global record_i
            print(f"Trial {record_i}: {score}")
            record_i += 1
            # wandb.log({'trial_score': score})
            return score

        study = optuna.create_study(direction="maximize")
        trial_num = 500
        study.optimize(object_func, n_trials=trial_num, n_jobs=1)

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
                if param in ["max_depth", "n_estimators"]:
                    # 产生整数序列
                    values = np.linspace(min_val, max_val, num, dtype=int).tolist()
                else:
                    # 产生浮点数序列
                    values = np.linspace(min_val, max_val, num).tolist()
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
            "max_depth": (1, 8, 8),
            "n_estimators": (10, 300, 75),
            "learning_rate": (0.0, 1.0, 1000),
        }

        param_grid = create_param_grid(param_ranges, num_combinations=100)
        print(param_grid)
        # assert 0, param_grid

        model = CatBoostRegressor(
            random_state=self.random_state,
            task_type=self.task_type,
            devices=self.devices,
        )

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
        with open(os.path.join(save_path, "CatBoost.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "CatBoost.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
