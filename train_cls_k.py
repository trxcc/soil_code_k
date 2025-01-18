import argparse
import datetime
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

import wandb
from classification_solver import get_model  
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
        

def run(args: SimpleNamespace):
    set_seed(args.seed)
    global results_dir, X
    data_file = args.data_file
    assert os.path.exists(data_file), f"data {data_file} not found"

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(
        base_path,
        "results",
        "k_classification",
        f"-seed{args.seed}-" + args.model,
    )
    model_dir = os.path.join(
        base_path, 
        "models", 
        "k_classification" + f"-seed{args.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})

    data = pd.read_excel(data_file, sheet_name=args.data_sheet).dropna()
    
    y = (data["Arrhenius equation"] == "Yes").astype(int).to_numpy()  # 转换为0/1标签
    
    data = data.drop(["Arrhenius equation"], axis=1)
    
    data = preprocess_data(
        data=data, 
        one_hot_indices=one_hot_columns, 
        remove_indices=excluded_columns
    )
    X = data.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    name = f"{args.model}-{args.seed}"
    ts_name = f"-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}"
    wandb.init(
        project="soil-k",
        group="k-classification",
        name=f"{args.seed}-" + name + ts_name,
        config=args.__dict__,
        job_type="train",
        mode='offline'
    )

    # # 获取并训练模型
    model = get_model(
        args.model,
        task_name="k_classification",
        seed=args.seed,
        model_dir=model_dir,
        results_dir=results_dir,
        n_classes=2,
        n_dim=X.shape[1]
    )
    
    model.fit(
        X_train=X_train,
        y_train=y_train,
        optimize_hyperparams=args.optimize_hyperparams,
        optimize_method=args.optimize_method,
    )
    
    model.save(model_dir)
    
    y_test_pred = model.predict(X_test)
    y_pred = model.predict(X)
    y_test_proba = model.predict_proba(X_test)[:, 1]  
    y_proba = model.predict_proba(X)[:, 1]
    
    test_accuracy = model.evaluate(X_test, y_test)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_avg_precision = average_precision_score(y_test, y_test_proba)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('test_roc.png')
    
    results_df = pd.DataFrame()
    results_df[f"fpr"] = pd.Series(fpr.squeeze().ravel())
    results_df[f"tpr"] = pd.Series(tpr.squeeze().ravel())
    results_df.to_excel(
        os.path.join(results_dir, f"Test-ROC-{args.model}-{args.seed}.xlsx")
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC-AUC Score: {test_roc_auc:.4f}")
    print(f"Test Average Precision Score: {test_avg_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test f1: {test_f1:.4f}")
    
    accuracy = model.evaluate(X, y)
    roc_auc = roc_auc_score(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_proba)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('all_roc.png')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
    
    results_df = pd.DataFrame()
    results_df[f"fpr"] = pd.Series(fpr.squeeze().ravel())
    results_df[f"tpr"] = pd.Series(tpr.squeeze().ravel())
    results_df.to_excel(
        os.path.join(results_dir, f"All-ROC-{args.model}-{args.seed}.xlsx")
    )
    
    results_df = pd.DataFrame()
    results_df[f"Arrhenius equation_true"] = pd.Series(y.ravel())
    results_df[f"Arrhenius equation_pred"] = pd.Series(y_pred.ravel())
    results_df.to_excel(
        os.path.join(results_dir, f"Arrhenius equation-{args.model}-{args.seed}.xlsx")
    )

    test_results_df = pd.DataFrame()
    test_results_df[f"Arrhenius equation_true"] = pd.Series(y_test.ravel())
    test_results_df[f"Arrhenius equation_pred"] = pd.Series(y_test_pred.ravel())
    test_results_df.to_excel(
        os.path.join(results_dir, f"Arrhenius equation-{args.model}-test-{args.seed}.xlsx")
    )
    
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        test_accuracy,
        "test_accuracy",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        test_roc_auc,
        "test_roc_auc",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        test_avg_precision,
        "test_avg_precision",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        test_recall,
        "test_recall",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        test_f1,
        "test_f1",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        accuracy,
        "all_accuracy",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        roc_auc,
        "all_roc_auc",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        avg_precision,
        "all_avg_precision",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        recall,
        "all_recall",
    )
    save_metric_to_csv(
        results_dir,
        args.optimize_method,
        args.model,
        args.seed,
        f1,
        "all_f1",
    )
    
    results_dir = os.path.join(
        base_path,
        "results",
        "k_classification" + "-all_data",
        f"-seed{args.seed}-" + args.model,
    )
    model_dir = os.path.join(
        base_path, 
        "models", 
        "k_classification" + "-all_data" + f"-seed{args.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.__dict__.update({"results_dir": results_dir, "model_dir": model_dir})
    
    model = get_model(
        args.model,
        task_name="k_classification",
        seed=args.seed,
        model_dir=model_dir,
        results_dir=results_dir,
        n_classes=2,
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
        # print(pred_data.columns)
        # pred_data = pred_data.drop(["Arrhenius equation"], axis=1)
        pred: np.ndarray = model.predict(pred_data)
        pred_data_copy = pred_data.copy()
        pred_data_copy["Arrhenius equation"] = pd.Series(pred.ravel(), index=pred_data_copy.index)
        pred_data_copy.dropna(axis=1, inplace=True)
        pred_data_copy.to_excel(os.path.join(
            results_dir, f'{sheet_name}-k_classification-{args.model}-pred.xlsx'
        ))
    
    predict_data('Sheet1')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-file", type=str, default="./train.xlsx")
    parser.add_argument("--data-sheet", type=str, default="Sheet2")
    parser.add_argument("--model", type=str, default="CatBoost")
    parser.add_argument("--optimize-hyperparams", action="store_true")
    parser.add_argument("--optimize-method", type=str, default="BayesOpt")
    args = parser.parse_args()

    run(args) 