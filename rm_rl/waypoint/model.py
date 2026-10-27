from __future__ import annotations

import torch
import torch.nn as nn


class WaypointMLP(nn.Module):
    def __init__(self, obs_dim, hidden=256, depth=2):
        super().__init__()
        layers = []
        width = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.depth = depth

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class WaypointClassifier(nn.Module):
    def __init__(self, obs_dim, n_classes, hidden=256, depth=2):
        super().__init__()
        layers = []
        width = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.n_classes = n_classes
        self.hidden = hidden
        self.depth = depth

    def forward(self, x):
        return self.net(x)
