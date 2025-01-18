import os

import pandas as pd

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def highlight_best_two(s, ascending=True):
    means = s.dropna().apply(lambda x: float(x.split("$\pm$")[0].strip()))

    sorted_means = means.sort_values(ascending=ascending)
    best_two = sorted_means.head(2).index

    new_s = s.copy()
    if len(best_two) > 0:
        new_s[best_two[0]] = f"\\textbf{{{new_s[best_two[0]]}}}"
    if len(best_two) > 1:
        new_s[best_two[1]] = f"\\underline{{{new_s[best_two[1]]}}}"

    return new_s


envs = ["k"]
env2result = {}

for env in envs:
    path = os.path.join(base_path, env + "-all_data")
    # path = os.path.join(base_path, "..", "k_results")
    results_csv = os.path.join(path, "all-final-results.csv")
    # results_csv = "all-ensemble-r2.csv"
    results = pd.read_csv(results_csv, index_col=0)  # .dropna(axis=1)
    results.columns = results.columns.astype(int)
    df_sorted = results.sort_index(axis=1)
    df_sorted.columns = df_sorted.columns.astype(str)
    results = df_sorted
    selected_columns =  [str(i) for i in range(1000, 5001, 1000)]
    results = results[[col for col in selected_columns if col in results.columns]]

    results.to_csv("k.csv")

    sorted_columns_0 = df_sorted.mean().sort_values(ascending=False).index
    max_columns = df_sorted.idxmax(axis=1)
    env2result[env] = results

result_df = pd.DataFrame()

for task_name, df in env2result.items():
    filtered_df = df
    print(filtered_df)

    means = filtered_df.mean(axis=1)
    std_devs = filtered_df.std(axis=1)

    result_df[task_name] = (
        means.round(3).astype(str) + " $\\pm$ " + std_devs.round(3).astype(str)
    )


from scipy.stats import ttest_ind


def calculate_p_values(df):
    means = df.mean(axis=1)
    best_algorithm = means.idxmax()

    p_values = {}

    for algorithm, row in df.iterrows():
        if algorithm != best_algorithm:
            stat, p_value = ttest_ind(row, df.loc[best_algorithm])
            p_values[algorithm] = p_value
        else:
            p_values[algorithm] = float("nan")

    return p_values


env2p = {}
for algo, df in env2result.items():
    env2p[algo] = calculate_p_values(df)
    print(algo)
    print()
    print(df)
    print()

for env, dic in env2p.items():
    for method, p in dic.items():
        result_df.loc[method, env] = f"{result_df.loc[method, env]} " + (
            "-" if p < 0.05 else "+"
        )

result_df = result_df.apply(highlight_best_two, ascending=False)
os.makedirs(f"./{envs[0]}_results/", exist_ok=True)
result_df.to_csv(f"./{envs[0]}_results/All-Main-Results-{envs[0]}-new-{len(selected_columns)}.csv")

print(result_df.to_latex())

print(sorted_columns_0)
print(max_columns)
