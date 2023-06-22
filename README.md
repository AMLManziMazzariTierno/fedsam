# Fedsam + Classifier Calibration with Virtual Representations

This repository is an extension of the following repo [Fedsam](https://github.com/debcaldarola/fedsam). It was added the implementation of Fed-CCVR, which is the solution described by the following paper
> Mi Luo, F. Chen, D. Hu, Y. Zhang, J. Liang, J. Feng [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data (https://arxiv.org/pdf/2106.05001.pdf), 35th Conference on Neural Information Processing Systems (NeurIPS 2021)

## Setup
### Environment
- Install conda environment (preferred): ```conda env create -f environment.yml```
- Install with pip (alternative): ```pip3 install -r requirements.txt```

### Weights and Biases
The code runs with WANDB. For setting up your profile, we refer you to the [quickstart documentation](https://docs.wandb.ai/quickstart). Insert your WANDB API KEY [here](https://github.com/debcaldarola/fedsam/blob/master/models/main.py#L24). WANDB MODE is set to "online" by default, switch to "offline" if no internet connection is available.

### Data
Execute the following code for setting up the datasets:
```bash
conda activate torch10
cd data
chmod +x setup_datasets.sh
./setup_datasets.sh
```

## Dataset

CIFAR-100
  * **Overview**: Image Dataset based on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 100 users with 500 images each. Different combinations are possible, following Dirichlet's distribution
  * **Task**: Image Classification over 100 classes

## Running experiments
Examples of commands for running the paper experiments (FedAvg/FedSAM/FedASAM w/ and w/o SWA) can be found in ```fedsam/paper_experiments```.
E.g. for CIFAR100 use the following command:
```bash
cd paper_experiments
chmod +x cifar100.sh
./cifar100.sh
```
