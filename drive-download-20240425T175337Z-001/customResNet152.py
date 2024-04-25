import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = models.resnet152(pretrained=True)

        # Replace the final fully connected layer of ResNet50 with a custom classifier
        # ResNet50's last layer input features: 2048
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # Forward pass through the modified ResNet50
        x = self.resnet(x)
        return x









