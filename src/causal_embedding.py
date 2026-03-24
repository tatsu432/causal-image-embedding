import torch
from torch import nn

from autoencoder import Decoder, Encoder


class DebiasedEmbeddingNet(nn.Module):
    def __init__(self, covariate_dim: int, covariate_image_dim: int, post_treatment_dim: int):
        super().__init__()
        self.covariate_image_encoder = Encoder(latent_dim=covariate_image_dim)
        self._post_treatment_encoder = Encoder(latent_dim=post_treatment_dim)
        self._decoder = Decoder(latent_dim=covariate_image_dim + post_treatment_dim)

        self._treatment_predictor = nn.Sequential(
            nn.Linear(covariate_dim + covariate_image_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self._post_treatment_predictor = nn.Sequential(
            nn.Linear(covariate_dim + 1, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, post_treatment_dim),
            nn.Sigmoid(),
        )

        self._outcome_predictor = nn.Sequential(
            nn.Linear(covariate_dim + 1 + covariate_image_dim + post_treatment_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1),
        )

    def forward(self, x, d, v, y):
        """
        x: covariate
        d: treatment
        v: image (we will create a covariate image and post-treatment from this)
        y: outcome
        """
        # Create a covariate image and post-treatment from the image
        x_v = self.covariate_image_encoder(v)
        p_v = self._post_treatment_encoder(v)

        # Concatenate the covariate image and post-treatment
        z = torch.cat([x_v, p_v], dim=-1)

        # Decode the concatenated covariate image and post-treatment
        hat_v = self._decoder(z)

        # Predict the treatment
        hat_d = self._treatment_predictor(torch.cat([x, x_v], dim=-1))
        hat_d = hat_d.squeeze(-1)

        # Predict the post-treatment
        hat_p_v = self._post_treatment_predictor(torch.cat([x, d.unsqueeze(-1)], dim=-1))

        # Predict the outcome
        hat_y = self._outcome_predictor(torch.cat([x, d.unsqueeze(-1), x_v, p_v], dim=-1))
        hat_y = hat_y.squeeze(-1)

        return x_v, p_v, hat_p_v, hat_d, hat_y, hat_v
