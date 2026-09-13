"""Mala CNN arhitektura za klasifikaciju zanra na log-mel spektrogramima.
Ulaz: (batch, 1, n_mels, vreme). Izlaz: (batch, n_zanrova) logiti.
"""
import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(self, n_genres: int, n_mels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, n_genres)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
