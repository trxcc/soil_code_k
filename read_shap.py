import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from solver import get_model

params = {
    "lines.linewidth": 1.5,
    "legend.fontsize": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
}
matplotlib.rcParams.update(params)

plt.rc("font", family="Times New Roman")

env2methodseed = {
    "em-241121": [("RandomForest", i) for i in [str(idx) for idx in range(1000, 5001, 1000)]]
    + [("CatBoost", i) for i in [str(idx) for idx in range(1000, 5001, 1000)]]
    + [("DeepForest", i) for i in [str(idx) for idx in range(1000, 5001, 1000)]]
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
shap_path = os.path.join(base_path, "shap_fig")
data_path = os.path.join(base_path, "shap_data")
os.makedirs(shap_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

shap.initjs()

occur_dict = {}

for env, method_seed_pairs in env2methodseed.items():

    all_columns = None

    def preprocess_data(
        data: pd.DataFrame,
        one_hot_indices: List[str] = [],
        remove_indices: List[str] = [],
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
        data = pd.concat([data, one_hot_encoded_data], axis=1)
        global all_columns
        if all_columns is None:
            all_columns = data.columns
        else:
            for column in all_columns:
                if column not in data.columns:
                    data[column] = 0
            data = data[all_columns]
        return data

    data_all = (
        pd.read_excel('train.xlsx', sheet_name='Sheet2').dropna().copy()
    )
    data_all = preprocess_data(data_all, remove_indices=["ID", "site"])

    X = data_all.drop(env, axis=1).to_numpy()
    y = data_all[env].to_numpy()

    for method, seed in method_seed_pairs:
        print(method, seed)
        path = os.path.join(results_dir, env + "-all_data", f"-seed{seed}-{method}")
        print(path)
        if not os.path.exists(os.path.join(path, "shap_values50.npy")):
            continue
        shap_values = np.load(os.path.join(path, "shap_values50.npy"))
        print(shap_values.shape)
        shap_mean_values = np.mean(shap_values, axis=0)
        print(shap_mean_values.shape)

        print(shap_values)
        print(shap_mean_values)

        new_shap_values = {}
        ecosystem_shap_values = []

        features = data_all.drop(env, axis=1).columns.to_list()

        if ecosystem_shap_values:
            new_shap_values["Ecosystem"] = np.mean(ecosystem_shap_values)

        sorted_features = sorted(
            new_shap_values, key=lambda x: abs(new_shap_values[x]), reverse=True
        )
        sorted_shap_values = np.array(
            [new_shap_values[feature] for feature in sorted_features]
        )

        df = pd.DataFrame(list(sorted_shap_values), index=sorted_features)
        df.to_excel(os.path.join(data_path, f"{method}-{seed}-{env}-ShapValues.xlsx"))

        fig, ax = plt.subplots(figsize=(15, 8))

        y_pos = np.arange(len(sorted_features))
        ax.barh(
            y_pos,
            sorted_shap_values,
            color=np.where(sorted_shap_values >= 0, "blue", "red"),
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)

        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importances of {method} on {env}")

        ax.invert_yaxis()

        plt.savefig(os.path.join(shap_path, f"{method}-{seed}-{env}.png"))
        plt.savefig(os.path.join(shap_path, f"{method}-{seed}-{env}.pdf"))
        plt.close()

sorted_items = sorted(occur_dict.items(), key=lambda item: item[1])

print("Sorted key-value pairs by ascending values:")
for key, value in sorted_items:
    print(f"{key}: {value + 2}")
