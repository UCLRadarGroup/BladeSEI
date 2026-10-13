import torch
import torch.nn as nn


class Inception1D(nn.Module):
    def __init__(self, in_channels,
                 ch1, ch3_reduce, ch3,
                 ch5_reduce, ch5,
                 pool_proj):
        super(Inception1D, self).__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch1),
            nn.ReLU(inplace=True)
        )

        # 1x1 to 3
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch3_reduce),
            nn.ReLU(inplace=True),

            nn.Conv1d(ch3_reduce, ch3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch3),
            nn.ReLU(inplace=True)
        )

        # 1x1 to 5
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch5_reduce),
            nn.ReLU(inplace=True),

            nn.Conv1d(ch5_reduce, ch5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(ch5),
            nn.ReLU(inplace=True)
        )

        # pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),

            nn.Conv1d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class Inception1DNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(Inception1DNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Inception blocks
        self.inception1 = Inception1D(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = Inception1D(256, 128, 128, 192, 32, 96, 64)
        self.inception3 = Inception1D(480, 192, 128, 208, 32, 48, 64)
        self.inception4 = Inception1D(512, 160, 144, 224, 40, 64, 64)
        self.inception5 = Inception1D(512, 336, 192, 400, 64, 176, 192)

        # Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1104, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.inception3(x)
        x = self.inception4(x)
        
        x = self.inception5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
