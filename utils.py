import random
from typing import List

import numpy as np
import pandas as pd
import torch

_all_columns = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True


def preprocess_data(
    data: pd.DataFrame, one_hot_indices: List[str] = [], remove_indices: List[str] = []
) -> pd.DataFrame:
    one_hot_encoded_data = None
    for one_hot_index in one_hot_indices:
        data[one_hot_index] = data[one_hot_index].str.lower()
        now_encoded_data = pd.get_dummies(
            data[one_hot_index], prefix=one_hot_index
        ).astype(float)
        if one_hot_encoded_data is None:
            one_hot_encoded_data = now_encoded_data
        else:
            one_hot_encoded_data = pd.concat(
                [one_hot_encoded_data, now_encoded_data], axis=1
            )
    data.drop(one_hot_indices + remove_indices, axis=1, inplace=True)
    if one_hot_encoded_data is not None:
        data = pd.concat([data, one_hot_encoded_data], axis=1)
    print(data.columns)
    global _all_columns
    if _all_columns is None:
        _all_columns = data.columns
    else:
        for column in _all_columns:
            if column not in data.columns:
                data[column] = 0
        data = data[_all_columns]
    print(data.shape)
    print(_all_columns)
    # with open(f"{target}_all_columns.txt", "w") as f:
    #     f.write(str(_all_columns.to_list()))
    return data
