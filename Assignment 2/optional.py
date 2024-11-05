import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define Model
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, grayscale=True):
        super(LeNet5, self).__init__()
        in_channels = 1 if grayscale else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 64, 1024),
            nn.Tanh(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# Initialize model and data loader
model = LeNet5()
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
sample_data, _ = next(iter(test_loader))
sample_image = sample_data[0].unsqueeze(0)

# Fix: Deconvolution Visualization Function
def deconv_visualize_feature(data, model, layer_index=0, feature_index=0):
    model.eval()
    data = data.cuda() if torch.cuda.is_available() else data
    with torch.no_grad():
        # Forward pass up to the selected layer
        activation = data
        for i, layer in enumerate(model.features):
            activation = layer(activation)
            if i == layer_index:
                break

        # Select and visualize the specific feature map
        selected_activation = activation[:, feature_index:feature_index + 1, :, :].clone()
        
        # Display the feature map
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(selected_activation.cpu().squeeze().numpy(), cmap="viridis")
        axs[0].set_title(f"Layer {layer_index} - Feature {feature_index} Activation")
        axs[0].axis("off")
        
        # Prepare deconvolution by iterating backwards through the layers
        deconv_output = selected_activation
        for i in range(layer_index, -1, -1):
            layer = model.features[i]
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.transpose(0, 1)
                deconv_output = F.conv_transpose2d(deconv_output, weight, stride=1, padding=2, output_padding=0)
            elif isinstance(layer, nn.MaxPool2d):
                # Perform unpooling by upsampling instead of MaxUnpool2d
                deconv_output = F.interpolate(deconv_output, scale_factor=2, mode='nearest')

        
        # Display deconvolution result
        deconv_output_np = deconv_output.cpu().squeeze().numpy()
        if deconv_output_np.ndim == 3:
            # Select the first channel
            image_to_plot = deconv_output_np[0]
        else:
            image_to_plot = deconv_output_np

        axs[1].imshow(image_to_plot, cmap="gray")
        axs[1].set_title(f"Deconv Output - Layer {layer_index} Feature {feature_index}")
        axs[1].axis("off")
        plt.show()


        # Display all channels in a grid
        num_channels = deconv_output_np.shape[0]
        fig, axs = plt.subplots(1, num_channels, figsize=(num_channels*2, 2))
        for idx in range(num_channels):
            axs[idx].imshow(deconv_output_np[idx], cmap="gray")
            axs[idx].axis("off")
        plt.show()


# Visualize deconvolution for the first few features
for feature_index in range(4):  # Visualize a few feature maps
    deconv_visualize_feature(sample_image, model, layer_index=0, feature_index=feature_index)
