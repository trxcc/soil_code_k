import math
import os
import pickle
from typing import Dict, Optional

import delu
import numpy as np
import optuna
# import wandb
import torch
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import RegressionModel


class FTTransformerModel(RegressionModel):
    def __init__(
        self,
        name: str = "FTTransformer",
        task_name: str = "",
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
    ):
        super().__init__(name)
        self.X_val = None
        self.y_val = None
        self.model = None
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_range = {"lr": (0, 0.01), "weight_decay": (0, 0.01)}
        # self.n_cont_features = n_dim
        # self.model = FTTransformer(
        #     n_cont_features=self.n_cont_features,
        #     cat_cardinalities=[],
        #     d_out=1,
        #     **FTTransformer.get_default_kwargs(),
        # ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        load_model: bool = False,
        model_path_if_load=None,
        optimize_hyperparams: bool = False,
        optimize_method=None,
        optimizing: bool = False,
        save_model: bool = False,
    ) -> None:

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=2023
        )
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        self.data = self._data_preprocess(X_train, y_train, X_val, y_val)

        self.n_cont_features = X_train.shape[1]
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=[],
            d_out=1,
            **FTTransformer.get_default_kwargs(),
        ).to(self.device)

        # if optimize_hyperparams:
        #     best_params = self.optimize_hyperparameters(
        #         X_train=X_train,
        #         y_train=y_train,
        #         model_args=self.param_range
        #     )

        #     if self.results_dir is not None:
        #         with open(os.path.join(self.results_dir, 'best_params.pkl'), 'wb+') as f:
        #             pickle.dump(obj=best_params, file=f)

        #     self.optimizer = torch.optim.AdamW(self.model.make_parameter_groups(), **best_params)

        # else:

        if load_model:
            assert model_path_if_load is not None
            self.load(model_path_if_load)
            self.is_trained = True
            return

        # optimizer = self.model.make_default_optimizer()
        self.optimizer = torch.optim.AdamW(
            self.model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5
        )

        self.apply_fn = lambda batch, model: model(
            batch["x_cont"], batch.get("x_cat")
        ).squeeze(-1)

        self.loss_fn = F.mse_loss

        n_epochs = 500
        patience = 16
        # n_epochs = 20
        # patience = 2

        # batch_size = 32 if not optimizing else 64
        batch_size = 128
        epoch_size = math.ceil(len(X_train) / batch_size)

        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        best = {
            "val": -math.inf,
            # "test": -math.inf,
            "epoch": -1,
        }

        print(f"Device: {self.device.type.upper()}")
        print("-" * 88 + "\n")
        timer.run()

        for epoch in range(n_epochs):
            losses = []
            for batch in tqdm(
                delu.iter_batches(self.data["train"], batch_size, shuffle=True),
                desc=f"Epoch {epoch}",
                total=epoch_size,
            ):
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.apply_fn(batch, self.model), batch["y"])
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            avg_loss = np.array(losses).flatten().mean()

            train_score = self.evaluate_model("train")
            val_score = self.evaluate_model("val")
            # test_score = self.evaluate_model("test")
            # print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

            # early_stopping.update(val_score)
            # if early_stopping.should_stop():
            #     break

            if val_score > best["val"]:
                print("🌸 New best epoch! 🌸")
                best = {
                    "val": val_score,
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                }
                if save_model:
                    self.save()
            print()

            # if not optimizing:
            #     wandb.log({
            #         "train/loss": avg_loss,
            #         "train/R2score": train_score,
            #         "val/R2score": val_score,
            #         "val/best_epoch": best["epoch"]
            #     })

        print("\n\nResult:")
        print(best["val"], best["epoch"])

        self.model.load_state_dict(best["model_state_dict"])
        if not optimizing:
            self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # 预处理数据
        X_test = self.preprocessing.transform(X_test)

        # 转换为torch.Tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # 创建 DataLoader
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # 初始化用于存放结果的列表
        predictions = []

        # 使用 torch.no_grad() 来停用梯度计算，节省计算和内存资源
        with torch.no_grad():
            for batch in test_loader:
                # 由于batch是一个列表，其中包含一个元素（我们的特征）
                batch_X = batch[0].to(self.device)

                # 进行预测
                batch_pred = self.model(batch_X, None).detach().cpu().numpy()

                # 将预测结果保存到列表中
                predictions.append(batch_pred)

        # 将分批预测的结果合并成一个数组
        predictions = np.concatenate(predictions, axis=0)

        # 反标准化处理
        return predictions * self.Y_std + self.Y_mean

    @torch.no_grad()
    def evaluate_model(self, part: str) -> float:
        self.model.eval()

        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    self.apply_fn(batch, self.model)
                    for batch in delu.iter_batches(self.data[part], eval_batch_size)
                ]
            )
            .cpu()
            .numpy()
        )
        y_true = self.data[part]["y"].cpu().numpy()
        score = r2_score(y_pred, y_true)
        # score = np.sqrt(mean_squared_error(y_true, y_pred))
        return score

    def optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:

        def object_func(trial):
            trial_args_dict = dict()
            # print(model_args.items())
            for key, (min_val, max_val) in model_args.items():
                trial_args_dict.update(
                    {key: trial.suggest_float(key, min_val, max_val)}
                )

            self.model = FTTransformer(
                n_cont_features=self.n_cont_features,
                cat_cardinalities=[],
                d_out=1,
                **FTTransformer.get_default_kwargs(),
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), **trial_args_dict
            )

            kf = KFold(n_splits=5, shuffle=True, random_state=2024)
            scores = []

            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # 训练模型
                self.fit(
                    X_train_fold,
                    y_train_fold,
                    optimize_hyperparams=False,
                    optimizing=True,
                    save_model=False,
                )

                # 评估模型
                score = self.evaluate(X_val_fold, y_val_fold)
                scores.append(score)

            # 计算平均得分
            score = np.mean(scores)
            # return average_score

            # score = cross_val_score(self, X_train, y_train, cv=5, scoring='r2').mean()
            # wandb.log({'trial_score': score})
            return score

        study = optuna.create_study(direction="maximize")
        trial_num = 100
        study.optimize(object_func, n_trials=trial_num)

        return study.best_trial.params

    def _data_preprocess(self, X_train, y_train, X_val, y_val):
        data_numpy = {
            "train": {"x_cont": X_train, "y": y_train},
            "val": {"x_cont": X_val, "y": y_val},
        }

        X_cont_train_numpy = data_numpy["train"]["x_cont"]
        self.preprocessing = QuantileTransformer(
            n_quantiles=max(min(len(X_cont_train_numpy) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        ).fit(X_cont_train_numpy)

        del X_cont_train_numpy

        for part in data_numpy:
            data_numpy[part]["x_cont"] = self.preprocessing.transform(
                data_numpy[part]["x_cont"]
            )

        self.Y_mean = data_numpy["train"]["y"].mean().item()
        self.Y_std = data_numpy["train"]["y"].std().item()
        for part in data_numpy:
            data_numpy[part]["y"] = (data_numpy[part]["y"] - self.Y_mean) / self.Y_std

        data = {
            part: {
                k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
                for k, v in data_numpy[part].items()
            }
            for part in data_numpy
        }

        if self.results_dir is not None:
            with open(os.path.join(self.results_dir, "preprocessing.pkl"), "wb+") as f:
                pickle.dump(
                    obj={
                        "preprocessing": self.preprocessing,
                        "y_mean": self.Y_mean,
                        "y_std": self.Y_std,
                    },
                    file=f,
                )

        return data

    def save(self, save_path=None) -> None:
        assert save_path is not None or self.model_dir is not None
        if save_path is None:
            save_path = self.model_dir
        self.model = self.model.to("cpu")
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, os.path.join(save_path, "FTTransformer.pt"))
        self.model = self.model.to(self.device)

    def load(self, model_dir, preprocessing, y_mean, y_std):
        self.preprocessing = preprocessing
        self.Y_mean = y_mean
        self.Y_std = y_std
        model_path = os.path.join(model_dir, "FTTransformer.pt")
        assert self.model is not None
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
