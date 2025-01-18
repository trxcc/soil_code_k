import argparse
import datetime
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import wandb
from solver import get_model
from utils import preprocess_data, set_seed

excluded_columns = ["ID", "site"]
one_hot_columns = []

results_dir = None 
X = None 

def save_metric_to_csv(
    results_dir, optimize_method, model_name, seed, metric_value, metric_name
):
    csv_path = os.path.join(
        results_dir,
        "..",
        ("GridSearch" if optimize_method == "GridSearch" else "")
        + f"{metric_name}.csv",
    )
    result = {"model_name": model_name, f"{seed}": metric_value}

    if not os.path.exists(csv_path):
        new_df = pd.DataFrame([result])
        new_df.to_csv(csv_path, index=False)
    else:
        existing_df = pd.read_csv(csv_path, header=0, index_col=0)
        updated_df = existing_df.copy()
        updated_df.loc[model_name, f"{seed}"] = metric_value
        updated_df.columns = updated_df.columns.astype(int)
        updated_df = updated_df.sort_index(axis=1)
        updated_df.to_csv(csv_path, index=True, mode="w")


def explain_model(regression_model):

    def f(X):
        nonlocal regression_model
        return regression_model.predict(X).flatten()

    global results_dir, X

    explainer = shap.KernelExplainer(f, X[:50, :])
    shap_values = explainer.shap_values(X[50, :], nsamples=500)
    np.save(arr=shap_values, file=os.path.join(results_dir, "shap_values.npy"))
    shap.initjs()
    shap_plot = shap.force_plot(explainer.expected_value, shap_values, X[50, :])
    shap.save_html(os.path.join(results_dir, "shap_single_predictions.html"), shap_plot)

    try:
        max_samples = min(500, X.shape[0])
        shap_values50 = explainer.shap_values(X[:max_samples, :], nsamples=max_samples)
        np.save(arr=shap_values50, file=os.path.join(results_dir, "shap_values50.npy"))
        shap_plot_all = shap.force_plot(
            explainer.expected_value, shap_values50, X[:, :]
        )
        shap.save_html(
            os.path.join(results_dir, "shap_many_predictions.html"), shap_plot_all
        )
    except:
        try:
            max_samples = min(500, X.shape[0])
            shap_values50 = explainer.shap_values(X[:max_samples, :], nsamples=max_samples)
            np.save(
                arr=shap_values50, file=os.path.join(results_dir, "shap_values50.npy")
            )
            shap_plot_all = shap.force_plot(
                explainer.expected_value, shap_values50, X[:, :]
            )
            shap.save_html(
                os.path.join(results_dir, "shap_many_predictions.html"), shap_plot_all
            )
        except:
            pass


def run(args: SimpleNamespace):
    set_seed(args.seed)
    global results_dir, X
    data_file = args.data_file
    assert os.path.exists(data_file), f"data {data_file} not found"

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(
        base_path,
        "results",
        args.target + "-all_data",
        f"-seed{args.seed}-" + args.model,
    )
    model_dir = os.path.join(
        base_path, "models", args.target + "all_data-only_train" + f"-seed{args.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})

    data = pd.read_excel(data_file, sheet_name=args.data_sheet).dropna()

    data = preprocess_data(
        data=data, one_hot_indices=one_hot_columns, remove_indices=excluded_columns
    )
    X = data.drop([args.target], axis=1).to_numpy()
    y = data[args.target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    name = f"{args.model}-{args.seed}"
    ts_name = f"-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}"
    wandb.init(
        project="soil-k",
        group=f"{args.target}-test",
        name=f"{args.seed}-"
        + name
        + ts_name
        + ("GridSearch" if args.optimize_method == "GridSearch" else ""),
        config=args.__dict__,
        job_type="train",
        mode='offline'
    )
    model = get_model(
        args.model,
        task_name=args.target,
        seed=args.seed,
        model_dir=model_dir,
        results_dir=results_dir,
        n_dim=X.shape[1]
    )
    model.fit(
        X_train=X_train,
        y_train=y_train,
        optimize_hyperparams=args.optimize_hyperparams,
        optimize_method=args.optimize_method,
    )
    model.save(model_dir)

    R2score = model.evaluate(X_test, y_test)
    y_test_pred = model.predict(X_test)
    R2score_all = model.evaluate(X, y)
    y_pred = model.predict(X)

    MSE_score_test = mean_squared_error(y_test, y_test_pred)
    MSE_score_all = mean_squared_error(y, y_pred)

    results_df = pd.DataFrame()
    results_df[f"{args.target}_true"] = pd.Series(y.ravel())
    results_df[f"{args.target}_pred"] = pd.Series(y_pred.ravel())
    results_df.to_excel(
        os.path.join(results_dir, f"{args.target}-{args.model}-{args.seed}.xlsx")
    )

    test_results_df = pd.DataFrame()
    test_results_df[f"{args.target}_true"] = pd.Series(y_test.ravel())
    test_results_df[f"{args.target}_pred"] = pd.Series(y_test_pred.ravel())
    test_results_df.to_excel(
        os.path.join(results_dir, f"{args.target}-{args.model}-test-{args.seed}.xlsx")
    )

    # wandb.log({'test/R2score': R2score, 'test/R2score_all': R2score_all, 'test/step': 1, 'test/MSE': MSE_score_test, 'test/MSE_all': MSE_score_all})
    print(R2score)
    print(R2score_all)

    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        R2score,
        "final-results",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        R2score_all,
        "all-final-results",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        MSE_score_test,
        "final-mse",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        MSE_score_all,
        "all-final-mse",
    )
    
    if not os.path.exists(
        os.path.join(results_dir, "shap_values50.npy")
    ):
        explain_model(model)

    model_dir = os.path.join(
        base_path,
        "models",
        args.target
        + "-all_data"
        + f"{args.seed}"
        + ("GridSearch" if args.optimize_method == "GridSearch" else ""),
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})

    model = get_model(
        args.model,
        task_name=args.target,
        seed=args.seed,
        model_dir=model_dir,
        results_dir=results_dir,
        n_dim=X.shape[1]
    )
    model.fit(
        X_train=X,
        y_train=y,
        optimize_hyperparams=args.optimize_hyperparams,
        optimize_method=args.optimize_method,
    )
    model.save(model_dir)
    
    def predict_data(sheet_name):
            
        pred_data = pd.read_excel('test_k.xlsx', sheet_name=sheet_name)
        excluded_columns.remove('site')
        pred_data = preprocess_data(
            data=pred_data, one_hot_indices=one_hot_columns, remove_indices=excluded_columns
        ).dropna()
        pred_data = pred_data.drop([args.target], axis=1)
        pred: np.ndarray = model.predict(pred_data)
        pred_data_copy = pred_data.copy()
        pred_data_copy[args.target] = pd.Series(pred.ravel(), index=pred_data_copy.index)
        pred_data_copy.dropna(axis=1, inplace=True)
        pred_data_copy.to_excel(os.path.join(
            results_dir, f'{sheet_name}-{args.target}-{args.model}-pred.xlsx'
        ))
    
    predict_data('Sheet1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-file", type=str, default="./train.xlsx")
    parser.add_argument("--data-sheet", type=str, default="Sheet1")
    parser.add_argument("--target", type=str, default="k")
    parser.add_argument("--model", type=str, default="RandomForest")
    parser.add_argument("--optimize-hyperparams", action="store_true")
    parser.add_argument("--optimize-method", type=str, default="BayesOpt")
    args = parser.parse_args()

    run(args)
