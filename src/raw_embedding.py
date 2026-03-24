import torch
from torch import nn
from tqdm import tqdm

from autoencoder import AutoEncoder


class RawEmbedding:
    def __init__(
        self,
        hidden_dim,
        train_loader,
        test_loader,
        batch_size=256,
        device="cpu",
        epochs=10,
        lr=1e-3,
        weight_decay=1e-5,
    ):
        self._hidden_dim = hidden_dim
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._batch_size = batch_size
        self._device = device
        self._epochs = epochs
        self._lr = lr
        self._weight_decay = weight_decay
        self._model = AutoEncoder(self._hidden_dim).to(self._device)

    def obtain_embeddings(self):
        # Define the optimizer, and loss function
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        criterion = nn.MSELoss()

        # Train the model
        for _epoch in tqdm(range(self._epochs)):
            self._model.train()
            loss_each_epoch = 0.0
            for x, _ in self._train_loader:
                x = x.to(self._device)
                reconstructed_x, _ = self._model(x)
                loss = criterion(reconstructed_x, x)
                optimizer.zero_grad()  # Clear the previous step graident
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the parameters by optimizer
                loss_each_epoch += loss.item() * x.size(0)
            # print(f"epoch {epoch+1}/{self._epochs} MSE: {loss_each_epoch/len(train_ds):.4f}")

        train_embeddings = self._compute_embeddings(self._train_loader)
        test_embeddings = self._compute_embeddings(self._test_loader)

        # Normalize the embeddings
        train_embeddings = (
            train_embeddings - train_embeddings.mean(dim=0, keepdim=True)
        ) / train_embeddings.std(dim=0, keepdim=True)
        test_embeddings = (
            test_embeddings - test_embeddings.mean(dim=0, keepdim=True)
        ) / test_embeddings.std(dim=0, keepdim=True)
        return train_embeddings, test_embeddings

    def _compute_embeddings(self, dataloader):
        self._model.eval()
        out = torch.zeros(len(dataloader.dataset), self._hidden_dim, device=self._device)
        idx = 0
        for x, _ in dataloader:
            batch_size = x.size(0)
            with torch.no_grad():
                z = self._model.encoder(x.to(self._device))
            out[idx : idx + batch_size] = z
            idx += batch_size
        return out.cpu()
