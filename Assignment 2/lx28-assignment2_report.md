# COMP 576 Assignment 2
**Lingyi Xu (lx28@rice.edu)**
**Nov ???th, 2024**


## 1. Visualizing a CNN with CIFAR10

### 1. a) CIFAR10 Dataset

In this implementation, we use PyTorch’s built-in data loading and preprocessing functions instead of using the `trainCifarStarterCode.py` script and manually downloaded dataset files. This approach uses PyTorch's `torchvision.datasets.CIFAR10` (as provided in the skeleton) to directly download and load the CIFAR-10 dataset, applying necessary transformations such as converting images to grayscale, scaling pixel values by dividing by 255 (through normalization), and standardizing with a mean of 0.5 and standard deviation of 0.5. This approach ensures efficient loading and preprocessing, maintaining consistency with the instructions without the need for manual dataset management.

Additionally, the labels were not manually one-hot encoded as PyTorch's CrossEntropyLoss handles categorical labels directly. This setup simplifies the data preparation pipeline while adhering to the preprocessing requirements for grayscale conversion and normalization.

### 1. b) Train LeNet5 on CIFAR10

Initially, I used a learning rate of 0.001 and the Adam optimizer as default settings. Here are the train/test accuracy and train loss plots generated:
<div style="display: flex; justify-content: space-around;">
    <img src="./figures/1_b_training_accuracy.png" alt="1_b_training_accuracy" style="width: 45%; height: auto;"/>
    <img src="./figures/1_b_training_loss.png" alt="1_b_training_loss" style="width: 45%; height: auto;"/>
</div>
<div style="display: flex; justify-content: space-around;">
    <img src="./figures/1_b_test_accuracy.png" alt="1_b_test_accuracy" style="width: 45%; height: auto;"/>
    <img src="./figures/1_b_test_loss.png" alt="1_b_test_loss" style="width: 45%; height: auto;"/>
</div>

The results obtained are as follows:
- Test Accuracy = 67.29
- Test Loss = 0.9764
- Training Accuracy = 73.662
- Training Loss = 0.75

Next, with the number of epochs set to 5, I experimented with different optimizers, learning rates, and momentum values. The results for train/test accuracy and train loss across various configurations are summarized in the table below:

| No. | Optimizer | Learning Rate | Momentum | Test Accuracy | Test Loss | Training Accuracy | Training Loss |
| --- | --------- | ------------- | -------- | ------------- | --------- | ----------------- | ------------- |
| 1   | Adam      | 0.001         | -        | 64.54         | 1.0411    | 66.85             | 0.9561        |
| 2   | SGD       | 0.001         | -        | 23.93         | 2.1566    | 22.886            | 2.1791        |
| 3   | Adam      | 0.01          | -        | 16.15         | 2.4999    | 16.15             | 2.6682        |
| 4   | Adam      | 0.005         | -        | 10.00         | 2.3034    | 10.08             | 2.3037        |
| 5   | SGD       | 0.01          | -        | 44.98         | 1.5635    | 43.63             | 1.5963        |
| 6   | SGD       | 0.005         | -        | 37.04         | 1.8221    | 35.19             | 1.8672        | 
| 7   | SGD       | 0.001         | 0.8      | 36.98         | 1.8251    | 35.11             | 1.8694        |
| 8   | SGD       | 0.001         | 0.5      | 28.09         | 2.0514    | 27.22             | 2.0659        |

Summary: Using the Adam optimizer with a learning rate of 0.001 provided the best overall model performance. This configuration achieved a balanced performance in terms of both training and test accuracy as well as loss, making it an ideal choice for the model.

### 1. c) Visualize the Trained Network
Using a learning rate of 0.001, 10 epochs, and the Adam optimizer, the visualization of the first convolutional layer’s weights is shown below:
<p align="center">
    <img src="./figures/1_c.png" alt="1_c" width="90%"/>
</p>

The statistics of the activations in the convolutional layers on test images are as follows:
```
Activation mean: -0.07476064562797546, variance: 0.03350016847252846
```
The mean value is close to 0, which indicates that the activations in the convolutional layer are relatively balanced. This distribution is typical and beneficial as it helps prevent gradient vanishing or exploding during training. The variance of 0.0335 indicates that the activations are spread within a narrow range, suggesting that the layer's responses to inputs have stabilized after training.

