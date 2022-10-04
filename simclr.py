from torch import nn
from torchvision import models


class Identity(nn.Module):
    def forward(self, x):
        return x


class SimCLR(nn.Module):
    def __init__(self, encoder="resnet18", projection_dim=128):
        super().__init__()

        encoders = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }

        if encoder not in encoders:
            raise RuntimeError(f"Unknown encoder received: {encoder}")

        # Build encoder
        encoder = encoders[encoder]()

        # Save output dimensionality
        encoder_out_dim = encoder.fc.in_features

        # Remove final FC layer
        encoder.fc = Identity()

        # Encoder
        self.encoder = encoder

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        iterable_type = type(x)
        h = iterable_type(self.encoder(x_i) for x_i in x)
        z = iterable_type(self.projection(h_i) for h_i in h)
        return h, z
