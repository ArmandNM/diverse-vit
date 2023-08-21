import pandas as pd
import numpy as np

from os import listdir
from os.path import join, isfile

from collections import defaultdict


LOGS_PATH = "/home/armand/repos/diverse-vit/results_logs"


def parse_csv(file_path: str):
    """Extract metrics grouped by epoch from a single experiment csv."""
    df = pd.read_csv(file_path)
    metric_names = list(df.columns[1:])

    num_epochs = df["epoch"].max()

    results = defaultdict(lambda: defaultdict(np.float64))

    complete_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        for metric in metric_names:
            entries = df[df["epoch"] == epoch][metric]
            if entries.isnull().all():
                complete_epochs = num_epochs - 1
                break

            assert (~entries.isnull()).sum() == 1, f"Multiple values for {metric} in epoch {epoch}"
            value = entries[~entries.isnull()].values[0]
            results[epoch][metric] = value

    if complete_epochs != num_epochs:
        del results[num_epochs]

    # Convert to dataframe
    items = []
    for epoch in results.keys():
        items.append([results[epoch][metric] for metric in metric_names])
    return pd.DataFrame(items, columns=metric_names, index=range(1, num_epochs + 1))


def parse_name(file_name: str):
    """Extract group name, run name and seed from the experiment fime_name.

    Experiment format: f"{run.group}__{run.name}__{config.seed}.csv"
    """
    splits = file_name[:-4].split("__")
    group_name = "__".join(splits[0:-2])
    run_name = "__".join(splits[-2:-1])
    seed = int(splits[-1])
    return group_name, run_name, seed


def get_experiment_summary(results: pd.DataFrame):
    """Compute experiment summary based on best head selection on validation data."""

    # Determine number of heads
    names = filter(lambda name: "Accuracy Head No" in name, results.columns)
    num_heads = max(map(lambda name: int(name.split("No")[1].split("/")[0]), names)) + 1

    # Compute ALL heads summary
    best_epoch = results["Accuracy (All Heads)/valOOD"].idxmax()
    best_accuracy = results.loc[best_epoch]["Accuracy (All Heads)/testOOD"]

    # Compute BEST head summary
    head_columns = [f"Accuracy Head No{head_idx}/valOOD" for head_idx in range(num_heads)]
    head_results = results[head_columns].to_numpy()
    best_head_indices = np.unravel_index(head_results.argmax(), head_results.shape)
    head_columns = [f"Accuracy Head No{head_idx}/testOOD" for head_idx in range(num_heads)]
    best_head_accuracy = results[head_columns].to_numpy()[best_head_indices]

    summary = {"best_all_accuracy": best_accuracy, "best_head_accuracy": best_head_accuracy}

    return summary


def compute_overview_all_runs(runs: dict):
    def print_results(sorted_results):
        for group_name, all_results, head_results in sorted_results:
            results_str = f"{group_name}\t\t"
            results_str += f"[ALL HEADS] {all_results[0]:.3f} +- {all_results[1]:.3f}\t"
            results_str += f"[BEST HEAD] {head_results[0]:.3f} +- {head_results[1]:.3f}"
            print(results_str)

    """Compute confidence intervals for each group and sort results."""
    results = []

    for group_name, run_summaries in runs.items():
        best_all_accuracies = np.array([s["best_all_accuracy"] for s in run_summaries])
        best_head_accuracies = np.array([s["best_head_accuracy"] for s in run_summaries])

        results.append(
            [
                group_name,
                (np.mean(best_all_accuracies), np.std(best_all_accuracies)),
                (np.mean(best_head_accuracies), np.std(best_head_accuracies)),
            ]
        )

        # f"{np.mean(avgs):.3f} +- {np.std(avgs):.3f}"

    # Sort results by best ALL accuracy
    results.sort(key=lambda x: x[1][0], reverse=True)
    print("Results sorted by ALL heads accuracy.\n")
    print_results(results)

    # Sort results by best HEAD accuracy
    print("\nResults sorted by BEST head accuracy.\n")
    results.sort(key=lambda x: x[2][0], reverse=True)
    print_results(results)


def main():
    runs = defaultdict(list)

    for file_name in listdir(LOGS_PATH):
        file_path = join(LOGS_PATH, file_name)

        group_name, run_name, seed = parse_name(file_name)

        if not (isfile(file_path) and file_name.endswith(".csv")):
            continue

        try:
            results = parse_csv(file_path)
            summary = get_experiment_summary(results)
            runs[group_name].append(summary)
        except Exception as e:
            print(f"Error while parsing {file_name}: {e}")

    compute_overview_all_runs(runs)


if __name__ == "__main__":
    main()
