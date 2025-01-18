import os
import pickle
import numpy as np
import optuna
from deepforest import CascadeForestClassifier
from sklearn.model_selection import cross_validate
from .base import ClassificationModel

class DeepForestModel(ClassificationModel):
    def __init__(
        self,
        name: str = "DeepForest",
        task_name: str = "",
        seed: int = 2024,
        n_classes: int = 2,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ) -> None:
        super().__init__(name)
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        # 默认参数设置
        self.max_depth = 6
        self.n_estimators = 100
        self.min_samples_split = 2
        
        self.random_state = seed
        
        # 初始化模型
        self.model = CascadeForestClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        # 参数优化范围
        self.param_range = {
            "max_depth": (3, 15),
            "n_estimators": (50, 500),
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

            self.model = CascadeForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
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
                trial_args_dict[key] = trial.suggest_int(key, min_val, max_val)

            model = CascadeForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
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
            
            # 计算综合评分
            final_score = (
                cv_results['test_accuracy'].mean() * 0.4 +
                cv_results['test_roc_auc'].mean() * 0.3 +
                cv_results['test_average_precision'].mean() * 0.3
            )
            return final_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2)
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
        else:
            raise NotImplementedError("仅支持贝叶斯优化方法。")

    def save(self, save_path: str) -> None:
        assert self.is_trained
        with open(os.path.join(save_path, "DeepForest_classifier.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_path: str) -> None:
        model_file = os.path.join(model_path, "DeepForest_classifier.pkl")
        assert os.path.exists(model_file)
        with open(model_file, "rb") as f:
            self.model = pickle.load(f) 