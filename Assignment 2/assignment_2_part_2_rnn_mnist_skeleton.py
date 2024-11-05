# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

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
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.001
try_cuda = True
seed = 1000
logging_interval = 10 # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28
# hidden_size = 128 
num_classes = 10

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

# Setting up data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# # plot one example
# print(train_dataset.data.size())     # (60000, 28, 28)
# print(train_dataset.targets.size())   # (60000)
# plt.imshow(train_dataset.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_dataset.targets[0])
# plt.show()

"""# Step 3: Creating the Model"""

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, rnn_type='RNN'):
        super(Net, self).__init__()

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid rnn_type. Choose from 'RNN', 'LSTM', or 'GRU'.")

        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        if isinstance(self.rnn, (nn.LSTM, nn.GRU)):
            r_out, _ = self.rnn(x, None)  # For LSTM/GRU, hidden state is a tuple
        else:
            r_out, _ = self.rnn(x)  # For RNN, hidden state is not a tuple

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

# model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='RNN')
# model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='LSTM')
# model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='GRU')

# if cuda:
#     model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=lr)

"""# Step 4: Train/Test"""

# Defining the test and trainig loops

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        output = model(data) # forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # # Log training metrics
        # if batch_idx % logging_interval == 0:
        #     print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
        #           f"Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    writer.add_scalar("Training Loss", avg_loss, epoch)
    writer.add_scalar("Training Accuracy", accuracy, epoch)
    print(f"Training set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)")
    return avg_loss, accuracy


def test(epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(-1, 28, 28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # if batch_idx % logging_interval == 0:
            #     print(f"Test Epoch: {epoch} [{batch_idx * len(data)}/{len(test_loader.dataset)}] "
            #           f"Batch Loss: {test_loss / ((batch_idx + 1) * len(data)):.6f}, Batch Accuracy: {100. * correct / ((batch_idx + 1) * len(data)):.2f}%")

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)")
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    return accuracy


# train_losses = []
# train_accuracies = []
# test_accuracies = []

# # Training loop
# for epoch in range(1, epochs + 1):
#     print(f"Starting Epoch {epoch}/{epochs}")
#     train_loss, train_accuracy = train(epoch)
#     test_accuracy = test(epoch)
    
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
#     test_accuracies.append(test_accuracy)
#     print(f"Epoch {epoch} Complete: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%\n")

# 3 b) different hidden_size
hidden_sizes = [64, 128, 256, 512]
results = {}

for hidden_size in hidden_sizes:
    print(f"\nTesting model with hidden size = {hidden_size}")
    
    # model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='RNN')
    # model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='LSTM')
    model = Net(input_size=INPUT_SIZE, hidden_size=hidden_size, num_classes=num_classes, rnn_type='GRU')
    
    if cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs} with hidden size {hidden_size}")
        train_loss, train_accuracy = train(epoch)
        test_accuracy = test(epoch)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch} Complete: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%\n")
    
    results[hidden_size] = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies
    }

plt.figure(figsize=(12, 6))
for hidden_size, result in results.items():
    plt.plot(result['test_accuracies'], label=f'Hidden Size {hidden_size}')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for hidden_size, result in results.items():
    plt.plot(result['train_losses'], label=f'Hidden Size {hidden_size}')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend()
plt.show()


writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""