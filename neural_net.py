# neural_net.py (注意力机制升级版)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        # <<< 核心改动：在残差块中加入SE模块 >>>
        self.se = SEBlock(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # <<< 核心改动：应用注意力机制 >>>
        x = self.se(x)
        x += residual
        x = F.relu(x)
        return x

class ExtendedConnectNet(nn.Module):
    # (这个类本身无需改动，因为它引用的ResBlock已经被我们升级了)
    def __init__(self, board_size=9, num_res_blocks=5, num_hidden=128, num_channels=20):
        super().__init__()
        self.board_size = board_size
        self.start_block = nn.Sequential(nn.Conv2d(num_channels, num_hidden, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(num_hidden), nn.ReLU())
        self.backbone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(nn.Conv2d(num_hidden, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(32 * board_size * board_size, board_size * board_size))
        self.value_head = nn.Sequential(nn.Conv2d(num_hidden, 3, kernel_size=1), nn.BatchNorm2d(3), nn.ReLU(),
                                        nn.Flatten(), nn.Linear(3 * board_size * board_size, 1), nn.Tanh())
    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone: x = res_block(x)
        policy = self.policy_head(x); value = self.value_head(x)
        return F.log_softmax(policy, dim=1), value