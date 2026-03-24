from torch import nn

from autoencoder import Decoder, Encoder


class NaiveEmbeddingNet(nn.Module):
    def __init__(self, covariate_dim: int, covariate_image_dim: int, post_treatment_dim: int):
        super().__init__()
        self.covariate_image_encoder = Encoder(latent_dim=covariate_image_dim)
        self._decoder = Decoder(latent_dim=covariate_image_dim)

    def forward(self, x, d, v, y):
        x_v = self.covariate_image_encoder(v)
        hat_v = self._decoder(x_v)
        return x_v, hat_v
