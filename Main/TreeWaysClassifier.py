from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Path 1: First Convolution Layer (32 kernels)
        self.conv_layers_path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )

        # Path 2: Eight Convolution Layers
        self.conv_layers_path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )

        # Path 3: Nine Convolution Layers
        self.conv_layers_path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),  # Adjust input size based on your architecture
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)  # Assuming binary classification (tumor or not)
        )

    def forward(self, x):
        # Path 1: First Convolution Layer
        x_path1 = self.conv1_path1(x)
        x_path1 = self.pool1_path1(x_path1)

        # Path 2: Eight Convolution Layers
        x_path2 = self.conv_layers_path2(x)

        # Path 3: Nine Convolution Layers
        x_path3 = self.conv_layers_path3(x)

        # Concatenate features from all three paths
        x = torch.cat((x_path1, x_path2, x_path3), dim=1)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.fc_layers(x)

        return x