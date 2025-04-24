import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 64)
        self.conv5 = conv_block(64, 128)
        self.conv6 = conv_block(128, 128)
        self.conv7 = conv_block(128, 128)
        self.conv8 = conv_block(128, 128)
        self.conv9 = conv_block(128, 128)
        self.conv10 = conv_block(128, 128)

        self.pool = nn.MaxPool2d(2, 2)

        # Input shape: (3, 200, 600) -> after 3 poolings -> (128, 25, 75)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25 * 75, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.conv2(x))     # -> (32, 100, 300)
        x = self.conv3(x)
        x = self.pool(self.conv4(x))     # -> (64, 50, 150)
        x = self.conv5(x)
        x = self.pool(self.conv6(x))     # -> (128, 25, 75)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.classifier(x)
        return x
