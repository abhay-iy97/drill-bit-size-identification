import torch.nn as nn
import torch

class LeNet5(nn.Module):
    def __init__(self):
        """
        LeNet5 model architecture intialization
        """
        super().__init__()
        self.featureExtraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1202064, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=9)
        )

    def forward(self, x):
        """Forward pass of CNN model

        Args:
            x (torch.tensor): Input to the model

        Returns:
            torch.tensor : Output of the model
        """
        x = self.featureExtraction(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
