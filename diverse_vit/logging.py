import torch
import csv
import wandb
import os


class CSVLogger:
    def __init__(self, csv_path, columns, mode="w"):
        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, metrics, epoch, split, log_wandb=True):
        print(f"Results for epoch {epoch} split {split}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        stats_dict = {f"{metric}/{split}": value for metric, value in metrics.items()}

        if log_wandb:
            wandb.log(stats_dict, step=epoch)

        stats_dict["epoch"] = epoch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def log_state_dict(run, state_dict, name, epoch, aliases, metadata, local_path):
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    filename = os.path.join(local_path, f"{name}_ep{epoch}.pth")
    torch.save(state_dict, filename)

    if "best" in aliases:
        torch.save(state_dict, os.path.join(local_path, f"{name}_best.pth"))

    if "latest" not in aliases:
        aliases.append("latest")

    artifact.add_file(filename)
    run.log_artifact(artifact, aliases=aliases)
