import os
import pickle

methods = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "DeepForest"]

for method in methods:
    for env in ["k_classification"]:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", env, f"-seed1-{method}"
        )
        if os.path.exists(os.path.join(path, "best_params.pkl")):
            print(env, method)
            print(path)
            with open(os.path.join(path, "best_params.pkl"), "rb+") as f:
                params = pickle.load(f)
                print(params)
