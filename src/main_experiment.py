import random
from collections.abc import Sized
from typing import cast

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from causal_embedding import DebiasedEmbeddingNet
from causal_inference import ATE, compute_ATE
from dataset import DatasetCausalInference, ObservedDataset
from naive_embedding import NaiveEmbeddingNet
from raw_embedding import RawEmbedding
from visualize import visualize_dataset

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DATASET CONFIGURATION ###
num_seeds = config.num_seeds
dim_covariate = config.dim_covariate
dim_covariate_image = config.dim_covariate_image
dim_post_treatment = config.dim_post_treatment

### AUTOENCODER CONFIGURATION ###
data_root = "./data"
batch_size_autoencoder = config.batch_size_autoencoder

### CAUSAL INFERENCE CONFIGURATION ###
# Specify the sample size for the training and test dataset
trainig_sample_size = config.trainig_sample_size
test_sample_size = config.test_sample_size

# Define the batch size for causal inference
batch_size_causal_embedding = config.batch_size_causal_embedding

# Define the learning rate, number of epochs, and weight decay for the embedding net
lr_embed = config.lr_embed
epochs_embed = config.epochs_embed
weight_decay_embed = config.weight_decay_embed

# Define the dimension of the embedding of the image covariate and post-treatment
dim_covariate_image_embed = config.dim_covariate_image_embed
dim_post_treatment_embed = config.dim_post_treatment_embed

dim_covariate_image_embed_naive = config.dim_covariate_image_embed_naive

# Define the dataframe for storing the results
df_result = pd.DataFrame(columns=["id", "estimator", "method", "train_err", "test_err"])

tfm = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)

train_dataset_no_transform = datasets.FashionMNIST(data_root, train=True, download=True)
test_dataset_no_transform = datasets.FashionMNIST(data_root, train=False, download=True)

train_dataset = Subset(train_dataset, range(config.n_train_fMNIST))
test_dataset = Subset(test_dataset, range(config.n_test_fMNIST))

# Create data loader for obtaining embedding of the raw images
train_loader = DataLoader(train_dataset, batch_size=batch_size_autoencoder, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_autoencoder, shuffle=False)

raw_embedding = RawEmbedding(
    hidden_dim=dim_covariate_image, train_loader=train_loader, test_loader=test_loader
)
train_embeddings, test_embeddings = raw_embedding.obtain_embeddings()

# Save and load the embeddings
torch.save((train_embeddings, test_embeddings), "fashion_mnist_embedding.pt")  # Save the embeddings
train_embeddings, test_embeddings = torch.load(
    "fashion_mnist_embedding.pt"
)  # Load the saved embeddings

# Create the dataset object for causal inference
dataset_ci = DatasetCausalInference(
    dim_covariate,
    dim_covariate_image,
    dim_post_treatment,
    train_embeddings,
    test_embeddings,
    train_dataset_no_transform,
    test_dataset_no_transform,
)

for seed in tqdm(range(num_seeds)):
    # Generate the training and test dataset for causal inference
    training_dataset_ci = dataset_ci.generate_dataset(trainig_sample_size, train=True)
    test_dataset_ci = dataset_ci.generate_dataset(test_sample_size, train=False)

    # Visualize the training and test dataset for causal inference
    # Left: original FashionMNIST images
    # Right: image included in the dataset, which includes the post-treatment variables' components
    if config.display_image:
        visualize_dataset(training_dataset_ci, max_size=3)
        visualize_dataset(test_dataset_ci, max_size=3)

    # Create the observed dataset for causal inference
    observed_train_dataset_ci = ObservedDataset(
        training_dataset_ci["covariate"],
        training_dataset_ci["treatment"],
        training_dataset_ci["post_treatment_image_dataset"],
        training_dataset_ci["outcome"],
    )
    observed_test_dataset_ci = ObservedDataset(
        test_dataset_ci["covariate"],
        test_dataset_ci["treatment"],
        test_dataset_ci["post_treatment_image_dataset"],
        test_dataset_ci["outcome"],
    )

    # Create the data loader for causal inference
    train_loader_ci = DataLoader(
        observed_train_dataset_ci, batch_size=batch_size_causal_embedding, shuffle=True
    )
    test_loader_ci = DataLoader(
        observed_test_dataset_ci, batch_size=batch_size_causal_embedding, shuffle=False
    )
    train_n_ci = len(cast(Sized, train_loader_ci.dataset))

    # Train the naive embedding net
    naive_embedding_net = NaiveEmbeddingNet(
        dim_covariate, dim_covariate_image_embed_naive, dim_post_treatment_embed
    )
    optimizer_naive_embed = torch.optim.Adam(
        naive_embedding_net.parameters(), lr=lr_embed, weight_decay=weight_decay_embed
    )
    mse_loss_naive = nn.MSELoss()

    for epoch in tqdm(range(epochs_embed), desc=f"Seed {seed} / {num_seeds}"):
        naive_embedding_net.train()
        loss_each_epoch = 0.0
        for batch in train_loader_ci:
            x, d, v, y = batch
            x_v, hat_v = naive_embedding_net(x, d, v, y)
            optimizer_naive_embed.zero_grad()
            loss_v = mse_loss_naive(hat_v, v)
            loss = loss_v
            loss.backward()
            optimizer_naive_embed.step()
            loss_each_epoch += loss.item() * x.size(0)
        if config.print_loss:
            print(f"Epoch {epoch + 1}/{epochs_embed} Loss: {loss_each_epoch / train_n_ci:.4f}")

    # Train the debiased embedding net
    debiased_embedding_net = DebiasedEmbeddingNet(
        dim_covariate, dim_covariate_image_embed, dim_post_treatment_embed
    )
    optimizer_embed = torch.optim.Adam(
        debiased_embedding_net.parameters(),
        lr=lr_embed,
        weight_decay=weight_decay_embed,
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in tqdm(range(epochs_embed)):
        debiased_embedding_net.train()
        loss_each_epoch = 0.0
        for batch in train_loader_ci:
            x, d, v, y = batch
            x_v, p_v, hat_p_v, hat_d, hat_y, hat_v = debiased_embedding_net(x, d, v, y)
            optimizer_embed.zero_grad()
            loss_v = mse_loss(hat_v, v)
            loss_d = bce_loss(hat_d, d)
            loss_p = mse_loss(hat_p_v, p_v)
            loss_y = mse_loss(hat_y, y)
            loss = loss_v + loss_d + loss_p + loss_y
            loss.backward()
            optimizer_embed.step()
            loss_each_epoch += loss.item() * x.size(0)
        if config.print_loss:
            print(f"Epoch {epoch + 1}/{epochs_embed} Loss: {loss_each_epoch / train_n_ci:.4f}")

    # Define the function to compute the embeddings given a dataloader and a model
    def compute_embeddings(dataloader, model, dim_covariate_image_embed):
        model.eval()
        out = torch.zeros(
            len(cast(Sized, dataloader.dataset)), dim_covariate_image_embed, device=device
        )
        idx = 0
        for batch in dataloader:
            x, _d, v, _y = batch
            batch_size = x.size(0)
            with torch.no_grad():
                z = model.covariate_image_encoder(v.to(device))
            out[idx : idx + batch_size] = z
            idx += batch_size
        return out.cpu()

    # Compute the image embeddings of the training and test dataset for the naive embedding net
    naive_image_embeddings_train = compute_embeddings(
        train_loader_ci, naive_embedding_net, dim_covariate_image_embed_naive
    )
    naive_image_embeddings_test = compute_embeddings(
        test_loader_ci, naive_embedding_net, dim_covariate_image_embed_naive
    )

    # Compute the image embeddings of the training and test dataset for the debiased embedding net
    debiased_image_embeddings_train = compute_embeddings(
        train_loader_ci, debiased_embedding_net, dim_covariate_image_embed
    )
    debiased_image_embeddings_test = compute_embeddings(
        test_loader_ci, debiased_embedding_net, dim_covariate_image_embed
    )

    # Define the function to compute the true ATE and the estimators
    def compute_ground_truth_ATE_and_estimators(
        dataset, naive_image_embeddings, debiased_image_embeddings
    ):
        # Compute the true ATE
        true_ATEs = compute_ATE(dataset, ate_type="true")
        true_ATE = true_ATEs.dr

        # Compute the biased ATE
        biased_ATEs = compute_ATE(dataset, ate_type="biased")

        # Compute the naive ATE
        naive_ATEs = compute_ATE(
            dataset,
            ate_type="learned_covariate_image",
            covariate_image=naive_image_embeddings,
        )

        # Compute the debiased ATE
        debiased_ATEs = compute_ATE(
            dataset,
            ate_type="learned_covariate_image",
            covariate_image=debiased_image_embeddings,
        )

        return ATE(true_ATE, biased_ATEs, naive_ATEs, debiased_ATEs)

    # Compute the true ATE and the estimates by each estimator and each method
    train_ATEs = compute_ground_truth_ATE_and_estimators(
        training_dataset_ci,
        naive_image_embeddings_train,
        debiased_image_embeddings_train,
    )
    test_ATEs = compute_ground_truth_ATE_and_estimators(
        test_dataset_ci, naive_image_embeddings_test, debiased_image_embeddings_test
    )

    # Create a row to add to the dataframe
    new_rows = [
        {
            "id": seed,
            "estimator": "regression",
            "method": "biased",
            "train_err": train_ATEs.biased_ATE.error_reg(train_ATEs.true_ATE),
            "test_err": test_ATEs.biased_ATE.error_reg(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "ipw",
            "method": "biased",
            "train_err": train_ATEs.biased_ATE.error_ipw(train_ATEs.true_ATE),
            "test_err": test_ATEs.biased_ATE.error_ipw(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "dr",
            "method": "biased",
            "train_err": train_ATEs.biased_ATE.error_dr(train_ATEs.true_ATE),
            "test_err": test_ATEs.biased_ATE.error_dr(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "regression",
            "method": "naive",
            "train_err": train_ATEs.naive_ATE.error_reg(train_ATEs.true_ATE),
            "test_err": test_ATEs.naive_ATE.error_reg(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "ipw",
            "method": "naive",
            "train_err": train_ATEs.naive_ATE.error_ipw(train_ATEs.true_ATE),
            "test_err": test_ATEs.naive_ATE.error_ipw(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "dr",
            "method": "naive",
            "train_err": train_ATEs.naive_ATE.error_dr(train_ATEs.true_ATE),
            "test_err": test_ATEs.naive_ATE.error_dr(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "regression",
            "method": "debiased",
            "train_err": train_ATEs.debiased_ATE.error_reg(train_ATEs.true_ATE),
            "test_err": test_ATEs.debiased_ATE.error_reg(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "ipw",
            "method": "debiased",
            "train_err": train_ATEs.debiased_ATE.error_ipw(train_ATEs.true_ATE),
            "test_err": test_ATEs.debiased_ATE.error_ipw(test_ATEs.true_ATE),
        },
        {
            "id": seed,
            "estimator": "dr",
            "method": "debiased",
            "train_err": train_ATEs.debiased_ATE.error_dr(train_ATEs.true_ATE),
            "test_err": test_ATEs.debiased_ATE.error_dr(test_ATEs.true_ATE),
        },
    ]

    # Add the row to the dataframe
    df_new = pd.DataFrame(new_rows)
    if config.print_result_per_seed:
        print(df_new)
    df_result = pd.concat([df_result, df_new], ignore_index=True)

# Save the resulting dataframe
df_result.to_pickle("df_result.pkl")
