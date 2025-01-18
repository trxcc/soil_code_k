from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score


class ClassificationModel(ABC):
    """分类问题的基础求解器

    返回一个分类模型。

    :param name: 分类模型的名称
    :param X_train: 训练数据的特征
    :param y_train: 训练数据的标签
    :param is_trained: 标识模型是否已训练的标志
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.X_train = None
        self.y_train = None
        self.is_trained = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_hyperparams: bool = False,
    ) -> None:
        """
        训练模型。

        :param X_train (array-like): 训练数据的特征
        :param y_train (array-like): 训练数据的标签
        """
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测

        给定X_test返回预测结果

        :params X_test (array-like): 测试数据的特征
        """
        pass

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """预测类别概率

        :params X_test (array-like): 测试数据的特征
        :return: 每个类别的预测概率
        """
        pass

    @abstractmethod
    def save(self, save_path: str) -> None:
        """保存模型

        :param save_path: 模型保存路径
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型

        :param model_path: 模型加载路径
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """评估模型性能

        返回测试集上的准确率分数

        :param X_test (array-like): 测试数据的特征
        :param y_test (array-like): 测试数据的标签
        :return: 准确率分数
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, model_args: dict
    ) -> dict:
        """优化模型的超参数

        返回经过优化的模型参数

        :param X_train (array-like): 训练数据的特征
        :param y_train (array-like): 训练数据的标签
        :param model_args: 包含待优化参数的字典
        :return: 优化后的参数字典
        """
        pass
