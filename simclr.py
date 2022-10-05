from torch import nn
from torchvision import models


class Identity(nn.Module):
    def forward(self, x):
        return x


class SimCLR(nn.Module):
    def __init__(self, encoder="resnet18", projection_features=128):
        super().__init__()

        encoders = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }

        if encoder not in encoders:
            raise RuntimeError(f"Unknown encoder received: {encoder}")

        # Build encoder
        self.encoder = encoders[encoder]()
        # Save encoder output dimensionality
        self.encoder_out_features = self.encoder.fc.in_features
        # Remove final FC layer
        self.encoder.fc = Identity()

        # Projection head
        self.projection = (
            nn.Sequential(
                nn.Linear(self.encoder_out_features, self.encoder_out_features),
                nn.ReLU(),
                nn.Linear(self.encoder_out_features, projection_features),
            )
            if projection_features is not None
            else None
        )

    def forward(self, x):
        iterable_type = type(x)
        h = iterable_type(self.encoder(x_i) for x_i in x)
        z = (
            iterable_type(self.projection(h_i) for h_i in h)
            if self.projection is not None
            else None
        )
        return h, z
