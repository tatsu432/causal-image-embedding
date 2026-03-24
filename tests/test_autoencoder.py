import torch

from autoencoder import AutoEncoder, Decoder, Encoder


def test_encoder_output_shape() -> None:
    latent = 8
    enc = Encoder(latent_dim=latent)
    x = torch.randn(4, 1, 28, 28)
    z = enc(x)
    assert z.shape == (4, latent)


def test_decoder_output_shape() -> None:
    latent = 8
    dec = Decoder(latent_dim=latent)
    z = torch.randn(4, latent)
    x_hat = dec(z)
    assert x_hat.shape == (4, 1, 28, 28)


def test_autoencoder_forward() -> None:
    latent = 12
    model = AutoEncoder(latent_dim=latent)
    x = torch.randn(2, 1, 28, 28)
    recon, z = model(x)
    assert z.shape == (2, latent)
    assert recon.shape == x.shape
