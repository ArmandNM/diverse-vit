# Training experiment launcher

import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import diverse_vit.training as training
import wandb

from diverse_vit.utils import set_random_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_experiment_name(config, custom_str=""):
    experiment_name = "DiverseViT__"
    experiment_name += f"{config.dataset}"
    experiment_name += f"__{config.optimizer}-{config.optimizer_params['lr']}"
    experiment_name += (
        f"__div-{config.diversification['weight']}-{config.diversification['normalization']}"
    )
    if config.scheduler == "MultiStepLR":
        experiment_name += f"__steps-{config.scheduler_params['milestones']}"

    if custom_str:
        experiment_name += f"__{custom_str}"

    return experiment_name


@hydra.main(config_path="../config", config_name="vision_diverse_mnist_cifar")
def main(config: DictConfig):
    with wandb.init(
        project="diverse-vit",
        entity="diverse-vit",
        config=OmegaConf.to_container(config),
        group=get_experiment_name(config),
    ) as run:
        # Retrieve used config
        config = wandb.config
        config.device = device

        # Log all code files
        run.log_code(".")

        # Set random seed
        set_random_seed(config.seed)

        training.main(config, run)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")
    main()
