# EECS 545 Fall 2020
import torch
import torchvision.models as models
from dataset import DogCatDataset
from train import train


def load_pretrained(num_classes=2):
    """
    Load a ResNet-18 model from `torchvision.models` with pre-trained weights. Freeze all the parameters besides the
    final layer by setting the flag `requires_grad` for each parameter to False. Replace the final fully connected layer
    with another fully connected layer with `num_classes` many output units.
    Inputs:
        - num_classes: int
    Returns:
        - model: PyTorch model
    """
    # TODO (part d): load a pre-trained ResNet-18 model

    return None


if __name__ == '__main__':
    config = {
        'dataset_path': 'data/images/dogs_vs_cats',
        'batch_size': 4,
        'ckpt_path': 'checkpoints/transfer',
        'plot_name': 'Transfer',
        'num_epoch': 10,
        'learning_rate': 1e-4,
    }
    dataset = DogCatDataset(config['batch_size'], config['dataset_path'])
    model = load_pretrained()
    train(config, dataset, model)
