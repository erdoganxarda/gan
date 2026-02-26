from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        conditional: bool = False,
        num_classes: int = 26,
        label_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.label_embed = nn.Embedding(num_classes, label_embed_dim) if conditional else None
        input_dim = latent_dim + (label_embed_dim if conditional else 0)
        self.proj = nn.Linear(input_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if labels is None:
                raise ValueError("labels must be provided when conditional=True")
            labels = labels.to(device=z.device, dtype=torch.long).view(-1)
            if labels.shape[0] != z.shape[0]:
                raise ValueError("labels batch size must match z batch size")
            assert self.label_embed is not None
            label_vec = self.label_embed(labels)
            z = torch.cat([z, label_vec], dim=1)
        x = self.proj(z)
        x = x.view(z.shape[0], 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        dropout: float = 0.2,
        conditional: bool = False,
        num_classes: int = 26,
        label_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.label_embed = nn.Embedding(num_classes, 28 * 28) if conditional else None
        in_channels = 2 if conditional else 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if labels is None:
                raise ValueError("labels must be provided when conditional=True")
            labels = labels.to(device=x.device, dtype=torch.long).view(-1)
            if labels.shape[0] != x.shape[0]:
                raise ValueError("labels batch size must match image batch size")
            assert self.label_embed is not None
            label_map = self.label_embed(labels).view(x.shape[0], 1, 28, 28)
            x = torch.cat([x, label_map], dim=1)
        feat = self.features(x)
        return self.classifier(feat)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
