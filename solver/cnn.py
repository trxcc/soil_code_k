# import wandb
import math
import os
import pickle

import delu
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import RegressionModel


class CNN(nn.Module):
    def __init__(self, input_length, pool_size, filters, activation, DenseNum):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, filters, kernel_size=2)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.conv2 = nn.Conv1d(filters, filters * 2, kernel_size=2)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.flatten = nn.Flatten()

        # 计算池化和卷积后的尺寸
        L_out1 = (input_length - 1) // pool_size
        L_out2 = (L_out1 - 1) // pool_size

        # 根据L_out2计算dense1的输入尺寸
        self.dense1 = nn.Linear(L_out2 * filters * 2, 64)
        self.dense2 = nn.Linear(64, 32)
        additional_layers = [nn.Linear(32, 16)] + [
            nn.Linear(16, 16) for _ in range(DenseNum - 3)
        ]
        self.additional_layers = nn.ModuleList(additional_layers)
        self.output = nn.Linear(16 if DenseNum > 2 else 32, 1)

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation '{activation}' is not supported.")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        for layer in self.additional_layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x


class CNNModel(RegressionModel):
    def __init__(
        self,
        name: str = "MLP",
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_range = {
            "lr": (0, 0.01),
            "weight_decay": (0, 0.01),
            "pool_size": (2, 5),
            "filters": (1, 4),
        }
        # self.n_cont_features = n_dim
        # self.model = CNN(
        #     input_length=self.n_cont_features,
        #     pool_size=3,
        #     filters=3,
        #     activation='relu',
        #     DenseNum=5,
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

        # if optimize_hyperparams:
        #     best_params = self.optimize_hyperparameters(
        #         X_train=X_train,
        #         y_train=y_train,
        #         model_args=self.param_range
        #     )

        #     if self.results_dir is not None:
        #         with open(os.path.join(self.results_dir, 'best_params.pkl'), 'wb+') as f:
        #             pickle.dump(obj=best_params, file=f)

        #     self.model = CNN(
        #         input_length=self.n_cont_features,
        #         pool_size=best_params['pool_size'],
        #         filters=best_params['filters'],
        #         activation='relu',
        #         DenseNum=5,
        #     ).to(self.device)

        #     self.optimizer = torch.optim.AdamW(
        #         self.model.parameters(),
        #         lr=best_params['lr'],
        #         weight_decay=best_params['weight_decay']
        #     )

        # else:
        self.model = CNN(
            input_length=self.n_cont_features,
            pool_size=3,
            filters=3,
            activation="relu",
            DenseNum=5,
        ).to(self.device)

        if load_model:
            assert model_path_if_load is not None
            self.load(model_path_if_load)
            self.is_trained = True
            return

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-5
        )

        self.apply_fn = lambda batch, model: model(batch["x_cont"]).squeeze(-1)

        self.loss_fn = F.mse_loss

        n_epochs = 500
        patience = 16
        # n_epochs = 20
        # patience = 2

        # batch_size = 32 if not optimizing else 64
        batch_size = 4 if not optimizing else 64
        epoch_size = math.ceil(len(X_train) / batch_size)

        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        best = {
            "val": -math.inf,
            # "test": -math.inf,
            "epoch": -1,
            "model": None,
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
            # wandb.log({
            #     "train/loss": avg_loss,
            #     "train/R2score": train_score,
            #     "val/R2score": val_score,
            #     "val/best_epoch": best["epoch"]
            # })

        print("\n\nResult:")
        print(best["val"], best["epoch"])

        self.model.load_state_dict(best["model_state_dict"])
        if not optimizing:
            self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = torch.from_numpy(self.preprocessing.transform(X_test)).to(
            device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            pred = self.model(X_test).detach().cpu().numpy()
        return pred * self.Y_std + self.Y_mean

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

    def _data_preprocess(self, X_train, y_train, X_val, y_val):
        data_numpy = {
            "train": {"x_cont": X_train, "y": y_train},
            "val": {"x_cont": X_val, "y": y_val},
        }

        X_cont_train_numpy = data_numpy["train"]["x_cont"]
        preprocessing = QuantileTransformer(
            n_quantiles=max(min(len(X_cont_train_numpy) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        ).fit(X_cont_train_numpy)
        del X_cont_train_numpy

        self.preprocessing = preprocessing

        if self.results_dir is not None:
            with open(os.path.join(self.results_dir, "preprocessing.pkl"), "wb+") as f:
                pickle.dump(obj=self.preprocessing, file=f)

        for part in data_numpy:
            data_numpy[part]["x_cont"] = preprocessing.transform(
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

        return data

    def optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:

        def object_func(trial):
            trial_args_dict = dict()
            # print(model_args.items())
            for key, (min_val, max_val) in model_args.items():
                if key in ["pool_size", "filters"]:
                    trial_args_dict.update(
                        {key: trial.suggest_float(key, min_val, max_val)}
                    )
                else:
                    trial_args_dict.update(
                        {key: trial.suggest_float(key, min_val, max_val)}
                    )

            self.model = CNN(
                input_length=self.n_cont_features,
                pool_size=trial_args_dict["pool_size"],
                filters=trial_args_dict["filters"],
                activation="relu",
                DenseNum=5,
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=trial_args_dict["lr"],
                weight_decay=trial_args_dict["weight_decay"],
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

    def save(self, save_path=None) -> None:
        assert save_path is not None or self.model_dir is not None
        if save_path is None:
            save_path = self.model_dir
        self.model = self.model.to("cpu")
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, os.path.join(save_path, "CNN.pt"))
        self.model = self.model.to(self.device)

    def load(self, model_dir, preprocessing, y_mean, y_std):
        self.preprocessing = preprocessing
        self.Y_mean = y_mean
        self.Y_std = y_std
        model_path = os.path.join(model_dir, "CNN.pt")
        assert self.model is not None
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
