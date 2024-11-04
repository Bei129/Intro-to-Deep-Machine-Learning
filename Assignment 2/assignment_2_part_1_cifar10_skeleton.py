# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

#hyperparameters
batch_size = 128
epochs = 10
lr = 0.001
optimizer_choice = "Adam"  # options: 'Adam', 'SGD', 'SGD+Momentum'
momentum_value = 0.5  # only used if SGD+Momentum is chosen
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

#otherum
logging_interval = 10 # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok = True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Downloading the CIFAR10 dataset

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 1024
padding = "same"
kernel_size = 5
verbose = False

# Calculate the side length of the final activation maps
final_length = 8

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                final_length * final_length * layer_2_n_filters * in_channels,
                fc_1_n_nodes,
            ),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(num_classes=num_classes, grayscale=grayscale)

if cuda:
    model.cuda()

# Set optimizer
if optimizer_choice == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_choice == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif optimizer_choice == "SGD+Momentum":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum_value)
else:
    raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")

"""# Step 4: Train/Test Loop"""

# Defining the test and trainig loops

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, probas = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = probas.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    # Calculate and log average training loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(
        f"Epoch {epoch}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
    )
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            logits, probas = model(data)
            test_loss += criterion(logits, target).item()
            pred = probas.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
    )
    writer.add_scalar("Test Loss", test_loss, epoch)
    writer.add_scalar("Test Accuracy", test_accuracy, epoch)


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)


# Visualize weights of the first convolutional layer
def visualize_weights(layer):
    weights = layer.weight.data.cpu().numpy()
    fig, axs = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axs.flatten()):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0, :, :], cmap="gray")
        ax.axis("off")
    plt.show()


# Visualize activations of the convolutional layer
def visualize_activations(data, model):
    with torch.no_grad():
        data = data.cuda() if cuda else data
        activation = model.features[0](data)
        activation = activation.cpu().numpy()

        # Calculate statistics of activations
        mean_activation = activation.mean()
        var_activation = activation.var()

        # Print activation statistics
        print(f"Activation mean: {mean_activation}, variance: {var_activation}")

# Select a batch of test data
data, _ = next(iter(test_loader))
visualize_weights(model.features[0])
visualize_activations(data, model)


writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""

