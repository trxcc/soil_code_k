import os
import pickle
import random

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score)

import wandb

from .base import RegressionModel


class LightGBMModel(RegressionModel):
    def __init__(
        self,
        name: str = "LightGBM",
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
            self.n_estimators = 534
            self.learning_rate = 0.12256790981239764
            self.reg_alpha = 17.671679461136065
            self.reg_lambda = 3.9897683484391537
        elif task_name == "GlcN":
            self.max_depth = 8
            self.n_estimators = 598
            self.learning_rate = 0.1242365803038221
            self.reg_alpha = 2.497878019946958
            self.reg_lambda = 0.24451254311624826
        elif task_name == "A_GN":
            self.max_depth = 18
            self.n_estimators = 191
            self.learning_rate = 0.1790449962443566
            self.reg_alpha = 0.4894051478807854
            self.reg_lambda = 0.953563157825089
        elif task_name == "F_GN":
            self.max_depth = 19
            self.n_estimators = 204
            self.learning_rate = 0.2507476745565035
            self.reg_alpha = 5.482009216301422
            self.reg_lambda = 18.808929943219745
        elif task_name == "k":
            self.max_depth = 12
            self.n_estimators = 423
            self.learning_rate = 0.7503858548207561
            self.reg_alpha = 2.751033221684443
            self.reg_lambda = 0.49772342205730313
        elif task_name == "k-240906":
            self.max_depth = 12
            self.n_estimators = 202
            self.learning_rate = 0.9997769274314039
            self.reg_alpha = 0.003269452092259928
            self.reg_lambda = 18.906454680566718
        elif task_name == "k-241004":
            self.max_depth = 15
            self.n_estimators = 886
            self.learning_rate = 0.9993148761193793
            self.reg_alpha = 3.908587753292715
            self.reg_lambda = 9.542439538415557
        elif task_name == "em-241004":
            self.max_depth = 14
            self.n_estimators = 921
            self.learning_rate = 0.11076643686322948
            self.reg_alpha = 0.008418829132728874
            self.reg_lambda = 0.32816447234981294
        elif task_name == "fPOC":
            self.max_depth = 11
            self.n_estimators = 74
            self.learning_rate = 0.018929679135278113
            self.reg_alpha = 1.2674405864045264
            self.reg_lambda = 2.4776120738071725
        elif task_name == "MAOC_N":
            self.max_depth = 20
            self.n_estimators = 839
            self.learning_rate = 0.3217105931382311
            self.reg_alpha = 7.626999576929737
            self.reg_lambda = 6.669403109902568
        elif task_name == "POC_N":
            self.max_depth = 17
            self.n_estimators = 450
            self.learning_rate = 0.4538026526962516
            self.reg_alpha = 5.049204575291155
            self.reg_lambda = 13.057092730036842
        else:
            self.max_depth = 6
            self.n_estimators = 534
            self.learning_rate = 0.12256790981239764
            self.reg_alpha = 17.671679461136065
            self.reg_lambda = 3.9897683484391537

        self.random_state = seed
        self.model = LGBMRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            boosting_type="goss",
            num_iterations=100,
            device="cpu",
            n_jobs=-1,
        )
        self.param_range = {
            "max_depth": (1, 20),
            "n_estimators": (10, 1000),
            "learning_rate": (0.0001, 1),
            "reg_alpha": (0.0001, 20),
            "reg_lambda": (0.0001, 20),
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

            print(best_params)

            self.model = LGBMRegressor(
                random_state=self.random_state,
                num_iterations=100,
                boosting_type="goss",
                device="cpu",
                **best_params
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
                if key in ["max_depth", "n_estimators"]:
                    trial_args_dict.update(
                        {key: trial.suggest_int(key, min_val, max_val)}
                    )
                else:
                    trial_args_dict.update(
                        {key: trial.suggest_float(key, min_val, max_val)}
                    )

            model = LGBMRegressor(
                random_state=self.random_state,
                num_iterations=100,
                device="cpu",
                boosting_type="goss",
                **trial_args_dict
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
            "max_depth": (1, 20, 10),
            "n_estimators": (10, 1000, 80),
            "learning_rate": (0.0, 1.0, 100),
            "reg_alpha": (0.0, 20.0, 40),
            "reg_lambda": (0.0, 20.0, 40),
        }

        param_grid = create_param_grid(param_ranges, num_combinations=100)
        print(param_grid)
        # assert 0, param_grid

        model = LGBMRegressor(
            random_state=self.random_state,
            num_iterations=100,
            boosting_type="goss",
            device="cpu",
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
        with open(os.path.join(save_path, "LightGBM.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "LightGBM.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
