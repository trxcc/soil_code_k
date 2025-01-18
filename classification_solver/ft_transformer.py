import math
import os
import pickle
from typing import Dict

import delu
import numpy as np
import torch
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import ClassificationModel


class FTTransformerModel(ClassificationModel):
    def __init__(
        self,
        name: str = "FTTransformer",
        task_name: str = "",
        seed: int = 2024,
        model_dir=None,
        results_dir=None,
        n_dim=None,
        n_classes=None,
    ):
        super().__init__(name)
        self.X_val = None
        self.y_val = None
        self.model = None
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_cont_features = n_dim
        self.n_classes = n_classes
        
        # 初始化模型
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=[],
            d_out=self.n_classes,  # 输出维度为类别数
            **FTTransformer.get_default_kwargs(),
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        load_model: bool = False,
        model_path_if_load=None,
        optimize_hyperparams: bool = False,
        optimize_method=None,
        save_model: bool = False,
    ) -> None:
        # 分割验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=2023
        )
        
        # 数据预处理
        self.data = self._data_preprocess(X_train, y_train, X_val, y_val)

        if load_model:
            assert model_path_if_load is not None
            self.load(model_path_if_load)
            self.is_trained = True
            return

        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            self.model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5
        )

        # 定义模型应用函数和损失函数
        self.apply_fn = lambda batch, model: model(batch["x_cont"], None)
        self.loss_fn = F.cross_entropy

        # 训练参数设置
        n_epochs = 500
        batch_size = 128
        epoch_size = math.ceil(len(X_train) / batch_size)

        # 训练追踪
        timer = delu.tools.Timer()
        best = {
            "val": -math.inf,
            "epoch": -1,
            "model_state_dict": None,
        }

        print(f"Device: {self.device.type.upper()}")
        print("-" * 88 + "\n")
        timer.run()

        # 训练循环
        for epoch in range(n_epochs):
            losses = []
            for batch in tqdm(
                delu.iter_batches(self.data["train"], batch_size, shuffle=True),
                desc=f"Epoch {epoch}",
                total=epoch_size,
            ):
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.apply_fn(batch, self.model), batch["y"].long())
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # 评估当前epoch
            val_score = self.evaluate_model("val")
            
            if val_score > best["val"]:
                print(f"🌸 新的最佳模型! 综合得分: {val_score:.4f} 🌸")
                best = {
                    "val": val_score,
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                }
                if save_model:
                    self.save()
            print(f"Epoch {epoch}: 验证集综合得分 = {val_score:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(best["model_state_dict"])
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # 预处理数据
        X_test = self.preprocessing.transform(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # 创建 DataLoader
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_X = batch[0].to(self.device)
                logits = self.model(batch_X, None)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.append(pred)
                
        return np.concatenate(predictions, axis=0)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        # 预处理数据
        X_test = self.preprocessing.transform(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # 创建 DataLoader
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_X = batch[0].to(self.device)
                logits = self.model(batch_X, None)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(probs)
                
        return np.concatenate(probabilities, axis=0)

    @torch.no_grad()
    def evaluate_model(self, part: str) -> float:
        self.model.eval()
        eval_batch_size = 8096
        
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        for batch in delu.iter_batches(self.data[part], eval_batch_size):
            logits = self.apply_fn(batch, self.model)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(batch["y"].cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_true_labels = np.array(all_true_labels)
        
        # 计算各个指标
        accuracy = accuracy_score(all_true_labels, all_predictions)
        
        # 对于二分类问题，只使用正类的概率
        if self.n_classes == 2:
            roc_auc = roc_auc_score(all_true_labels, all_probabilities[:, 1])
            average_precision = average_precision_score(all_true_labels, all_probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(all_true_labels, all_probabilities, multi_class='ovr')
            average_precision = average_precision_score(all_true_labels, all_probabilities, average='macro')
        
        # 使用与 LightGBM 相同的加权方式
        final_score = (
            accuracy * 0.4 +
            roc_auc * 0.3 +
            average_precision * 0.3
        )
        
        return final_score

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

        if self.results_dir is not None:
            with open(os.path.join(self.results_dir, "preprocessing.pkl"), "wb+") as f:
                pickle.dump(obj=self.preprocessing, file=f)

        for part in data_numpy:
            data_numpy[part]["x_cont"] = self.preprocessing.transform(
                data_numpy[part]["x_cont"]
            )

        data = {
            part: {
                k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
                for k, v in data_numpy[part].items()
            }
            for part in data_numpy
        }

        return data

    def save(self, save_path=None) -> None:
        assert save_path is not None or self.model_dir is not None
        if save_path is None:
            save_path = self.model_dir
        self.model = self.model.to("cpu")
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, os.path.join(save_path, "FTTransformer_classifier.pt"))
        self.model = self.model.to(self.device)

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "FTTransformer_classifier.pt")
        assert os.path.exists(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device) 