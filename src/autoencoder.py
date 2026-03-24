from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Output size: B, 32, 14, 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Output size: B, 64, 7, 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),  # Output size: B, latent_dim
        )

    def forward(self, x):
        return self._net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # Reshape to B, 64, 7, 7
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),  # Output size: B, 32, 14, 14
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),  # Output size: B, 1, 28, 28
        )

    def forward(self, z):
        return self._net(z)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z
