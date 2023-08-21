# DiverseViT

Implementation of our paper:


> [Learning Diverse Features in Vision Transformers for Improved Generalization](https://openreview.net/forum?id=cowlR3XdWV)
> 
> Armand Mihai Nicolicioiu, Andrei Liviu Nicolicioiu, Bogdan Alexe, Damien Teney
> 
> ICML Workshop on Spurious Correlations, Invariance, and Stability ([SCIS](https://sites.google.com/view/scis-workshop-23/home)) 2023:

## Installation

```bash
# Create new Python environment
conda create -n diverse-vit python=3.10
conda activate diverse-vit

# Install required libraries
pip install -r requirements.txt
```

## Running
Run the training loop with default parameters:
```bash
cd scripts/
export PYTHONPATH=..
python main.py
```
The parameters are configurable using [Hydra](https://hydra.cc/) and can be overriden from the CLI.
```bash
python main.py diversification.weight=100 optimizer_params.lr=0.001 seed=42 <extra_args>
```
Make sure to also replace the `data_path`, `logging_path`, and `checkpoints_path` from the [default config](https://github.com/ArmandNM/diverse-vit/blob/main/config/vision_diverse_mnist_cifar.yaml) with your own.

## Experiments
The reproduce our best results for both *Empirical Risk Minimization* and *Diversification* use the following overrides:

- ERM: `python main.py diversification.weight=0 optimizer_params.lr=0.0001`
- Diversification: `python main.py diversification.weight=100 optimizer_params.lr=0.001`

We use the following configuration for the Vision Transformer:

```yaml
model: 'DiverseViT'
model_params:
  image_size: [64, 32]
  patch_size: 4
  num_classes: 2
  channels: 3
  dim: 64
  depth: 6
  heads: 4
  mlp_dim: 128
```

We provide checkpoints to evaluate the model by running:

```bash
CHECKPOINT_PATH=/home/armand/repos/diverse-vit/checkpoints/ckpt_diverse_ep37.pth
python main.py checkpoint=$CHECKPOINT_PATH
```

Additionaly, all the logs for 10 seeds for each experiment are in [results_logs](https://github.com/ArmandNM/diverse-vit/tree/main/results_logs). To report the results summary, run the following script (after making sure you replace `LOGS_PATH` with your own):
```bash
cd scripts/
python reporting.py
```

The output should look like below:


```text
Results sorted by ALL heads accuracy.

DiverseViT__adam-0.001__div-100.0   [ALL HEADS] 0.646 +- 0.017  [BEST HEAD] 0.704 +- 0.017
DiverseViT__adam-0.0001__div-0.0    [ALL HEADS] 0.627 +- 0.010  [BEST HEAD] 0.645 +- 0.030

Results sorted by BEST head accuracy.

DiverseViT__adam-0.001__div-100.0   [ALL HEADS] 0.646 +- 0.017  [BEST HEAD] 0.704 +- 0.017
DiverseViT__adam-0.0001__div-0.0    [ALL HEADS] 0.627 +- 0.010  [BEST HEAD] 0.645 +- 0.030
```

## Citation
If our project is relevant for your research, please cite it using:

```bibtex
@incollection{nicolicioiu2023diversevit,
    title = {Learning Diverse Features in Vision Transformers for Improved Generalization},
    author = {Nicolicioiu, Armand Mihai and Nicolicioiu, Andrei Liviu and Alexe, Bogdan and Teney, Damien },
    booktitle = {ICML Workshop on Spurious Correlations, Invariance and Stability (SCIS)},
    year = {2023}
}
```
