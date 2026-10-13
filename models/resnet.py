import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        
class ResNet12_1D(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet12_1D, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1
        self.layer1 = nn.Sequential(
            BasicBlock1D(64, 64),
            BasicBlock1D(64, 64)
        )

        # Stage 2
        self.layer2 = nn.Sequential(
            BasicBlock1D(64, 128, stride=2),
            BasicBlock1D(128, 128)
        )

        # Stage 3
        self.layer3 = nn.Sequential(
            BasicBlock1D(128, 256, stride=2),
            BasicBlock1D(256, 256)
        )

        # Stage 4
        self.layer4 = nn.Sequential(
            BasicBlock1D(256, 256),
            BasicBlock1D(256, 256)
        )

        # Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
