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

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

#hyperparameters
batch_size = 128
epochs = 10
lr = 0.001
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

# downloading the cifar10 dataset


transform=[insert-code: create transforms, will need to include turning data grayscale]

train_dataset = [insert-code: download and transform cifar10 training data]
test_dataset = [insert-code: download and transform cifar10 test data]

train_loader = [insert-code: create train data loader]
test_loader = [insert-code: create test data loader]

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = [insert-code]
layer_2_n_filters = [insert-code]
fc_1_n_nodes = [insert-code]
padding="same"
kernel_size = 5
verbose = False

# calculating the side length of the final activation maps
final_length = [insert-code: calculate the dimension of the output of the \
            CNN stage before the MLP layers given the previous 2 convolutional layers \
                with the current padding setting, kernel size and maxpooling
            ]

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

            [insert-code]
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length*final_length*layer_2_n_filters*in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )


    def forward(self, x):
        x = [insert-code: send input through convolutional layers]
        x = [insert-code: send input through MLP layers]
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = [insert-code]

if cuda:
    model.cuda()

optimizer = [insert-code: USE AN ADAM OPTIMIZER]

"""# Step 4: Train/Test Loop"""

# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        [insert-code: move data to GPU]

        optimizer.zero_grad()
        logits,probas = model(data) # forward
        
        [insert-code: finish training loop and logging metrics]

    # 
    [insert-code: Log model parameters to TensorBoard at every epoch]

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(size_average = False)
    for data, target in test_loader:
        [insert-code: move data to GPU]

        logits,probas  = model(data)
        
        
        [insert-code: finish testing loop and logging metrics]


[insert-code: running test and training over epoch]

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""

