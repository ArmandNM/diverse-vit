import numpy as np
import torch
import torch.nn as nn
import tqdm
from ignite.metrics import Accuracy, Loss
from torchmetrics import MeanMetric
import wandb

import os
import pickle

import diverse_vit.utils as utils
from diverse_vit.logging import log_state_dict
from diverse_vit.losses import input_gradient_loss


def train_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    input_name,
    target_name,
    device,
    scheduler=None,
    diversity_weight=0.0,
    normalization="none",
):
    running_accuracy = Accuracy()
    running_loss = Loss(criterion)
    running_loss_diversity = MeanMetric().to(device)

    for data in tqdm.tqdm(data_loader):
        # Extract necessary data
        inputs, labels = data[input_name], data[target_name]

        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, extras = model(inputs, return_io=True)
        _, preds = torch.max(outputs, dim=-1)

        loss = criterion(outputs, labels)

        loss_diversity = input_gradient_loss(
            outputs,
            extras["inputs"],
            extras["outputs"],
            n_heads=model.vit.last_attn_num_heads,
            normalize_grads=normalization,
        )

        if diversity_weight > 0.0:
            loss += diversity_weight * loss_diversity

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss.update((outputs, labels))
        running_accuracy.update((outputs, labels))
        running_loss_diversity.update(loss_diversity)

    # Update learning rate if scheduler exists
    if scheduler is not None:
        scheduler.step()

    return {
        "Accuracy": running_accuracy.compute(),
        "Loss": running_loss.compute(),
        "Loss_diversity": running_loss_diversity.compute().item(),
    }


def evaluate_epoch(
    model, data_loader, criterion, input_name, target_name, device, compute_diversity=False
):
    running_accuracy = Accuracy()
    running_loss = Loss(criterion)
    if compute_diversity:
        running_loss_diversity = MeanMetric().to(device)

    for data in tqdm.tqdm(data_loader):
        # Extract necessary data
        inputs, labels = data[input_name], data[target_name]

        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        if compute_diversity:
            outputs, extras = model(inputs, return_io=True)
            _, preds = torch.max(outputs, dim=1)

            loss_diversity = input_gradient_loss(
                outputs, extras["inputs"], extras["outputs"], n_heads=model.vit.last_attn_num_heads
            )
        else:
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

        # Track metrics
        running_loss.update((outputs, labels))
        running_accuracy.update((preds, labels))
        if compute_diversity:
            running_loss_diversity.update(loss_diversity)

    metrics = {
        "Accuracy": running_accuracy.compute(),
        "Loss": running_loss.compute(),
    }

    if compute_diversity:
        metrics["Loss_diversity"] = running_loss_diversity.compute().item()

    return metrics

def evaluate_epoch_head_combinations(
    model,
    data_loader,
    split_name,
    epoch_id,
    criterion,
    input_name,
    target_name,
    device,
    logger,
):
    def rename_metrics(metrics, head_indices):
        renamed_metrics = {}

        if head_indices == "all":
            head_info = "(All Heads)"
        elif "Head" in head_indices:
            head_info = head_indices
        else:
            head_indices = [str(int(h_i)) for h_i in head_indices]
            head_info = f'(Heads {"".join(head_indices)})'

        for key, value in metrics.items():
            renamed_metrics[f"{key} {head_info}"] = value

        return renamed_metrics

    print("Evaluate using all heads")
    n_heads = model.vit.last_attn_num_heads
    model.vit.head_indices = None
    metrics_all_heads = evaluate_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        input_name=input_name,
        target_name=target_name,
        device=device,
        compute_diversity=True,
    )
    metrics_all_heads = rename_metrics(metrics_all_heads, head_indices="all")
    logger.log(metrics_all_heads, epoch_id, split=split_name)

    best_head_acc = 0.0
    best_head_metrics = {}
    for head_i in range(n_heads):
        head_indices = np.zeros(n_heads, dtype=bool)
        head_indices[head_i] = True
        head_indices = list(head_indices)
        model.vit.head_indices = head_indices
        print(f"Evaluate Head {head_indices}")
        metrics = evaluate_epoch(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            input_name=input_name,
            target_name=target_name,
            device=device,
            compute_diversity=False,
        )
        if metrics["Accuracy"] > best_head_acc:
            best_head_acc = metrics["Accuracy"]
            best_head_metrics = metrics
        metrics = rename_metrics(metrics, head_indices=f"Head No{head_i}")
        logger.log(metrics, epoch_id, split=split_name)

    # Plot the best head (at each iteration it could be a different head)
    best_head_metrics = rename_metrics(best_head_metrics, head_indices=f"Head Best")
    logger.log(best_head_metrics, epoch_id, split=split_name)


def run_training(
    model, data_loaders, criterion, optimizer, scheduler, input_name, target_name, config, logger
):
    best_acc = 0

    for epoch_id in range(1, config.num_epochs + 1):
        print(f"Training epoch {epoch_id}...")
        model.train()
        model.vit.head_indices = None
        metrics = train_epoch(
            model=model,
            data_loader=data_loaders["trainID"],
            criterion=criterion,
            optimizer=optimizer,
            input_name=input_name,
            target_name=target_name,
            device=config.device,
            scheduler=scheduler,
            diversity_weight=config.diversification["weight"],
            normalization=config.diversification["normalization"],
        )
        logger.log(metrics, epoch_id, split="trainID")

        run_evaluation(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            input_name=input_name,
            target_name=target_name,
            config=config,
            epoch_id=epoch_id,
            logger=logger,
        )

        aliases = ["latest", f"ep{epoch_id}"]
        metadata = {"epoch": epoch_id}

        log_state_dict(
            run=wandb.run,
            state_dict=model.state_dict(),
            name=f"ckpt_{wandb.run.name}",
            epoch=epoch_id,
            aliases=aliases,
            metadata=metadata,
            local_path=config.checkpoints_path,
        )

        logger.flush()

        print("_" * 50)


def run_evaluation(
    model, data_loaders, criterion, input_name, target_name, config, epoch_id, logger
):
    print("Evaluating in distribution")
    model.eval()
    evaluate_epoch_head_combinations(
        model=model,
        data_loader=data_loaders["valID"],
        split_name="valID",
        epoch_id=epoch_id,
        criterion=criterion,
        input_name=input_name,
        target_name=target_name,
        device=config.device,
        logger=logger,
    )

    print("Evaluating out of distribution")
    model.eval()
    evaluate_epoch_head_combinations(
        model=model,
        data_loader=data_loaders["valOOD"],
        split_name="valOOD",
        epoch_id=epoch_id,
        criterion=criterion,
        input_name=input_name,
        target_name=target_name,
        device=config.device,
        logger=logger,
    )

    print("Testing in distribution")
    model.eval()
    evaluate_epoch_head_combinations(
        model=model,
        data_loader=data_loaders["testID"],
        split_name="testID",
        epoch_id=epoch_id,
        criterion=criterion,
        input_name=input_name,
        target_name=target_name,
        device=config.device,
        logger=logger,
    )

    print("Testing out of distribution")
    model.eval()
    evaluate_epoch_head_combinations(
        model=model,
        data_loader=data_loaders["testOOD"],
        split_name="testOOD",
        epoch_id=epoch_id,
        criterion=criterion,
        input_name=input_name,
        target_name=target_name,
        device=config.device,
        logger=logger,
    )


def main(config={}, run=None):
    print("Running Diverse.")
    if config.checkpoint:
        print(f"Evaluation only. Using checkpoint: {config.checkpoint}")

    # Get dataset
    if not os.path.exists(config.dataset_params["data_path"]):
        os.makedirs(config.dataset_params["data_path"])
    datasets_cache_path = os.path.join(config.dataset_params["data_path"], "datasets.pkl")
    if os.path.exists(datasets_cache_path):
        with open(datasets_cache_path, "rb") as f:
            datasets = pickle.load(f)
    else:
        datasets = utils.get_dataset(name=config.dataset, config=config.dataset_params, run=run)
        with open(datasets_cache_path, "wb") as f:
            pickle.dump(datasets, f)

    data_loaders = {}
    batch_size = config.dataset_params["batch_size"]
    for name, dataset in datasets.items():
        data_loaders[name] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    # Get model
    model = utils.get_model(name=config.model, config=config.model_params)
    model.to(config.device)
    wandb.watch(model, log="all", log_freq=500)

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Get optimizer
    optimizer = utils.get_optimizer(
        name=config.optimizer,
        model_params=model.parameters(),
        config=config.optimizer_params,
    )

    # Get scheduler
    scheduler = utils.get_scheduler(
        name=config.scheduler, optimizer=optimizer, config=config.scheduler_params
    )

    # Get logger
    if not os.path.exists(config.logging_path):
        os.makedirs(config.logging_path)
    csv_path = os.path.join(config.logging_path, f"{run.group}__{run.name}__{config.seed}.csv")
    logger = utils.get_logger(csv_path=csv_path, num_heads=model.vit.last_attn_num_heads)

    if not config.checkpoint:
        run_training(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            input_name="img",
            target_name=config.target_name,
            config=config,
            logger=logger,
        )
    else:
        model.load_state_dict(torch.load(config.checkpoint), strict=False)
        run_evaluation(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            input_name="img",
            target_name=config.target_name,
            config=config,
            epoch_id=0,
            logger=logger,
        )

    logger.close()