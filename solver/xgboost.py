import os
import pickle
import random

import numpy as np
import optuna
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score)
from tqdm import tqdm
from xgboost import XGBRegressor

import wandb

from .base import RegressionModel


class XGBoostModel(RegressionModel):
    def __init__(
        self,
        name: str = "XGBoost",
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
            self.max_depth = 4
            self.n_estimators = 581
            self.learning_rate = 0.03462005934762355
            self.reg_alpha = 28.78591418316689
            self.reg_lambda = 12.273204335865751
            self.gamma = 19.874890564189798
        elif task_name == "GlcN":
            self.max_depth = 2
            self.n_estimators = 380
            self.learning_rate = 0.2654613982496461
            self.reg_alpha = 25.02664882379207
            self.reg_lambda = 36.541427731343035
            self.gamma = 39.45617396060322
        elif task_name == "A_GN":
            self.max_depth = 12
            self.n_estimators = 419
            self.learning_rate = 0.35334100881085256
            self.reg_alpha = 16.569043583120774
            self.reg_lambda = 16.793939597469834
            self.gamma = 0.09946530944762344
        elif task_name == "F_GN":
            self.max_depth = 20
            self.n_estimators = 839
            self.learning_rate = 0.6694034688481609
            self.reg_alpha = 0.9934218970070283
            self.reg_lambda = 21.679049567383576
            self.gamma = 0.32800793088269575
        elif task_name == "k":
            self.max_depth = 17
            self.n_estimators = 23
            self.learning_rate = 0.7282820052847035
            self.reg_alpha = 1.0593145461716895
            self.reg_lambda = 25.35884728252959
            self.gamma = 0.002478282856105553
        elif task_name == "k-240906":
            self.max_depth = 9
            self.n_estimators = 54
            self.learning_rate = 0.7632772714334224
            self.reg_alpha = 1.1428965801425477
            self.reg_lambda = 32.791286747716626
            self.gamma = 0.004574290920563662
        elif task_name == "k-241004":
            self.max_depth = 14
            self.n_estimators = 511
            self.learning_rate = 0.3719718362462099
            self.reg_alpha = 39.445289226506816
            self.reg_lambda = 27.388529592971235
            self.gamma = 13.438223308920676
        elif task_name == "em-241004":
            self.max_depth = 17
            self.n_estimators = 23
            self.learning_rate = 0.487058974460685
            self.reg_alpha = 13.750115054007377
            self.reg_lambda = 29.261989491086425
            self.gamma = 47.896361252982814
        elif task_name == "fPOC":
            self.max_depth = 19
            self.n_estimators = 860
            self.learning_rate = 0.5369009750086958
            self.reg_alpha = 0.927714484620682
            self.reg_lambda = 19.116688843035472
            self.gamma = 5.1030954431433635
        elif task_name == "MAOC_N":
            self.max_depth = 3
            self.n_estimators = 963
            self.learning_rate = 0.4479903790701641
            self.reg_alpha = 0.0160783336137037
            self.reg_lambda = 21.135464135108172
            self.gamma = 1.3234354361345888
        elif task_name == "POC_N":
            self.max_depth = 1
            self.n_estimators = 568
            self.learning_rate = 0.9478811278560136
            self.reg_alpha = 4.910068477402197
            self.reg_lambda = 26.876768155614567
            self.gamma = 0.5416851272715906
        else:
            self.max_depth = 17
            self.n_estimators = 855
            self.learning_rate = 0.19549990038696874
            self.reg_alpha = 1.282646777890526
            self.reg_lambda = 5.140178768001431
            self.gamma = 0.11585691459411045
        self.random_state = seed
        self.tree_method = "hist"
        self.model = XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.param_range = {
            "max_depth": (1, 20),
            "n_estimators": (10, 1000),
            "learning_rate": (0, 1),
            "reg_alpha": (0, 50),
            "reg_lambda": (0, 50),
            "gamma": (0, 50),
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X=None,
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

            self.model = XGBRegressor(
                random_state=self.random_state,
                tree_method=self.tree_method,
                **best_params
            )

        # import pandas as pd

        # from sklearn.metrics import r2_score as R2
        # from sklearn.metrics import mean_squared_error as MSE
        # from sklearn.metrics import explained_variance_score as EVS
        # from sklearn.metrics import mean_absolute_error as MAE

        # from sklearn.model_selection import cross_val_score as CVS
        # from matplotlib import pyplot as plt
        # import optuna
        # from sklearn.model_selection import train_test_split as TTS
        # import numpy as np

        # targetName = 'k'

        # # assert 0, (X_train, y_train)
        # trainPath = r'数据删除.xlsx'
        # dataBase = pd.read_excel(trainPath)
        # dataBase.dropna(inplace=True)
        # dataBaseCopy = dataBase.copy()

        # #%%======================================================
        # #取出特征和标签
        # X = dataBase.drop(['ID', targetName, 'Site', 'k', '95CILow', '95CIHigh', 'p'], axis=1)
        # XName = X.columns
        # Y = dataBase[targetName]

        # X_train, xTest, y_train, yTest = TTS(X, Y, test_size=0.3, random_state=2023)

        # bestParams = {'max_depth': 17,
        # 'n_estimators': 855,
        # 'learning_rate': 0.19549990038696874,
        # 'reg_alpha': 1.282646777890526,
        # 'reg_lambda': 5.140178768001431,
        # 'gamma': 0.11585691459411045,
        # 'random_state': 673}

        # self.model = XGBRegressor(max_depth=bestParams['max_depth'],
        #                     n_estimators=bestParams['n_estimators'],
        #                     learning_rate=bestParams['learning_rate'],
        #                     reg_alpha=bestParams['reg_alpha'],
        #                     reg_lambda=bestParams['reg_lambda'],
        #                     gamma=bestParams['gamma'],
        #                     random_state=bestParams['random_state'],
        #                     n_jobs=-1)

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
                if key in ["max_depth", "n_estimators", "random_state"]:
                    trial_args_dict.update(
                        {key: trial.suggest_int(key, min_val, max_val)}
                    )
                else:
                    trial_args_dict.update(
                        {key: trial.suggest_float(key, min_val, max_val)}
                    )

            model = XGBRegressor(
                random_state=self.random_state,
                tree_method=self.tree_method,
                **trial_args_dict
            )
            score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
            # wandb.log({'trial_score': score})
            return score

        study = optuna.create_study(direction="maximize")
        trial_num = 100
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
            param_grid = list(tqdm(ParameterGrid(grid)))

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
            "reg_alpha": (0.0, 50.0, 20),
            "reg_lambda": (0.0, 50.0, 20),
            "gamma": (0.0, 50.0, 20),
        }

        param_grid = create_param_grid(param_ranges, num_combinations=100)
        print(param_grid)
        # assert 0, param_grid

        model = XGBRegressor(
            random_state=self.random_state,
            tree_method=self.tree_method,
        )

        # 定义评分函数
        scorer = make_scorer(r2_score)

        # 创建 GridSearchCV 对象
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            cv=5,
            n_jobs=20,  # 使用所有可用的CPU核心
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
        with open(os.path.join(save_path, "XGBoost.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "XGBoost.pkl")
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
