import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms


class ObservedDataset(torch.utils.data.Dataset):
    def __init__(self, covariate, treatment, dataset, outcome):
        self._covariate = covariate
        self._treatment = treatment
        self._dataset = dataset
        self._outcome = outcome

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        tensor_image, _label = self._dataset[index]
        return (
            self._covariate[index],
            self._treatment[index],
            tensor_image,
            self._outcome[index],
        )


class PostTreatmentDataset(torch.utils.data.Dataset):
    def __init__(self, raw_image_dataset, post_treatment, max_size=10):
        self._raw_image_dataset = raw_image_dataset
        self._post_treatment = post_treatment
        self._max_size = max_size
        self._heart_icon = Image.open("images/white_heart.png").convert("L")
        self._star_icon = Image.open("images/white_star.png").convert("L")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self._raw_image_dataset)

    def __getitem__(self, index):
        original_image, label = self._raw_image_dataset[index]
        pt = self._post_treatment[index]

        # Convert original_image to tensor if it's a PIL Image
        if isinstance(original_image, Image.Image):
            original_image = self.transform(original_image)

        # Select icon based on first element of post_treatment
        if pt[0] > 0.5:
            icon = self._star_icon
        else:
            icon = self._heart_icon

        # Calculate size and highlight from post_treatment
        scaling_factor = 1 - pt[1]
        scaling_factor *= 3
        alpha = 1 - pt[2]

        if pt[3] > 0.5:
            put_left = True
        else:
            put_left = False

        if pt[4] > 0.5:
            put_top = True
        else:
            put_top = False

        # Ensure icon size is at least 1 pixel
        icon_size = max(1, int(self._max_size * scaling_factor))

        # Apply icon overlay if conditions are met
        if icon_size > 0 and alpha > 0:
            # Resize icon and convert to tensor
            icon_resized = icon.resize((icon_size, icon_size))
            icon_tensor = self.transform(icon_resized)

            # Ensure original_image is a tensor
            if not isinstance(original_image, torch.Tensor):
                original_image = self.transform(original_image)

            if put_left and put_top:
                # Blend icon onto top-left corner
                blended = (
                    alpha * icon_tensor + (1 - alpha) * original_image[:, :icon_size, :icon_size]
                )
                blended = torch.clamp(blended, 0, 1)
                # Update original_image with blended region
                original_image[:, :icon_size, :icon_size] = blended
            elif put_left and not put_top:
                # Blend icon onto buttom-left corner
                blended = (
                    alpha * icon_tensor + (1 - alpha) * original_image[:, -icon_size:, :icon_size]
                )
                blended = torch.clamp(blended, 0, 1)
                # Update original_image with blended region
                original_image[:, -icon_size:, :icon_size] = blended
            elif not put_left and put_top:
                # Blend icon onto top-right corner
                blended = (
                    alpha * icon_tensor + (1 - alpha) * original_image[:, :icon_size, -icon_size:]
                )
                blended = torch.clamp(blended, 0, 1)
                # Update original_image with blended region
                original_image[:, :icon_size, -icon_size:] = blended
            else:
                # Blend icon onto buttom-right corner
                blended = (
                    alpha * icon_tensor + (1 - alpha) * original_image[:, -icon_size:, -icon_size:]
                )
                blended = torch.clamp(blended, 0, 1)
                # Update original_image with blended region
                original_image[:, -icon_size:, -icon_size:] = blended

        return original_image, label


class DatasetCausalInference:
    def __init__(
        self,
        dim_covariate: int,
        dim_covariate_image: int,
        dim_post_treatment: int,
        train_embeddings: torch.Tensor,
        test_embeddings: torch.Tensor,
        train_dataset_no_transform: torch.utils.data.Dataset,
        test_dataset_no_transform: torch.utils.data.Dataset,
    ):
        self._dim_covariate = dim_covariate  # Dimension of the covariate
        self._dim_covariate_image = dim_covariate_image  # Dimension of the covariate in image
        self._dim_post_treatment = (
            dim_post_treatment  # Dimension of the post-treatment covariate in image
        )
        self._dim_outcome = 1  # Dimension of the outcome
        self._train_embeddings = train_embeddings
        self._test_embeddings = test_embeddings
        self._num_train = train_embeddings.shape[0]
        self._num_test = test_embeddings.shape[0]
        self._train_dataset_no_transform = train_dataset_no_transform
        self._test_dataset_no_transform = test_dataset_no_transform

        # Coefficients for generating the treatment
        self._coef1_treatment = torch.randn(dim_covariate)
        self._coef2_treatment = torch.randn(dim_covariate_image)

        # Coefficients for generating the post-treatment covariate
        self._coef1_post_treatment = torch.randn(dim_covariate, dim_post_treatment)
        self._coef2_post_treatment = torch.randn(dim_post_treatment)

        # Coefficients for generating the outcome
        self._coef1_outcome = torch.randn(dim_covariate)
        self._coef2_outcome = torch.randn(1)
        self._coef3_outcome = torch.randn(dim_covariate_image)
        self._coef4_outcome = torch.randn(dim_post_treatment)

    def _generate_covariate(self, sample_size: int):
        """
        X_i \sim N(0, I)
        """
        covariate = torch.randn(sample_size, self._dim_covariate)
        return covariate

    def _generate_image_index(self, sample_size: int, train: bool):
        """
        We sample the image index uniformly from the training set if train is True, otherwise from the test set.
        """
        if train:
            image_index = torch.randint(0, self._num_train, (sample_size,))
        else:
            image_index = torch.randint(0, self._num_test, (sample_size,))
        return image_index

    def _generate_raw_image(self, image_index: torch.Tensor, train: bool):
        """
        Create a raw image dataset from the image index.
        """
        idx_list = image_index.int().tolist()
        if train:
            raw_image_dataset = Subset(self._train_dataset_no_transform, idx_list)
        else:
            raw_image_dataset = Subset(self._test_dataset_no_transform, idx_list)
        return raw_image_dataset

    def _generate_covariate_image(self, sample_size: int, image_index: torch.Tensor, train: bool):
        """
        We use the embeddings of the training set if train is True, otherwise we use the embeddings of the test set.
        Then, we add Gaussian noise to the covariate X_i^V of the image.
        """
        if train:
            covariate_image = self._train_embeddings[image_index]
        else:
            covariate_image = self._test_embeddings[image_index]
        covariate_image += torch.normal(mean=0, std=0.1, size=covariate_image.shape)
        return covariate_image

    def _generate_treatment(self, covariate: torch.Tensor, covariate_image: torch.Tensor):
        """
        D_i \sim Bernoulli(p)
        p = \sigma(f_D(X_i, X_i^V))
        f_D(X_i, X_i^V) = X_i \coef_1 + X_i^V \coef_2
        """
        p = (
            covariate @ self._coef1_treatment / self._dim_covariate
            + covariate_image @ self._coef2_treatment / self._dim_covariate_image
        )
        p = torch.sigmoid(p)
        treatment = torch.bernoulli(p)
        return treatment

    def _generate_post_treatment(self, covariate: torch.Tensor, treatment: torch.Tensor):
        """
        P_i^V = \sigma(f_P(X_i, D_i) + \epsilon_i^P)
        f_P(X_i, D_i) = X_i \coef_1 + D_i \coef_2
        """
        post_treatment = (
            covariate @ self._coef1_post_treatment / self._dim_covariate
            + treatment.view(-1, 1) @ self._coef2_post_treatment.view(1, -1)
        )
        post_treatment += torch.normal(mean=0, std=0.1, size=post_treatment.shape)
        post_treatment = torch.sigmoid(post_treatment)
        return post_treatment

    def _generate_image_post_treatment(
        self, raw_image_dataset: torch.utils.data.Dataset, post_treatment: torch.Tensor
    ):
        """
        Create a dataset of images that are post-treated. We add either of two icons in the image depending on the value of the post-treatment.
        The value of the post-treatment also affects the size of the icon by reducing the size of the icon but it does not increase the size of the icon
        because the size of the icon is already at the maximum. We also allow the icon to be less highlighted depending on the value of the post-treatment.
        """
        # TODO: Implement this function
        post_treatment_image_dataset = PostTreatmentDataset(raw_image_dataset, post_treatment)
        return post_treatment_image_dataset

    def _generate_outcome(
        self,
        covariate: torch.Tensor,
        treatment: torch.Tensor,
        covariate_image: torch.Tensor,
        post_treatment: torch.Tensor,
    ):
        """
        Y_i = f_Y(X_i, D_i, X_i^V, P_i^V) + \epsilon_i^Y
        f_Y(X_i, D_i, X_i^V, P_i^V) = X_i \coef_1 + D_i \coef_2 + X_i^V \coef_3 + P_i^V \coef_4
        """
        outcome = (
            covariate @ self._coef1_outcome / self._dim_covariate
            + treatment * self._coef2_outcome
            + covariate_image @ self._coef3_outcome / self._dim_covariate_image
            + post_treatment @ self._coef4_outcome / self._dim_post_treatment
        )
        outcome += torch.normal(mean=0, std=0.1, size=outcome.shape)
        return outcome

    def generate_dataset(self, sample_size: int, train: bool):
        # Generate the covariate X_i from a standard normal distribution
        covariate = self._generate_covariate(sample_size)

        # Generate the image index
        image_index = self._generate_image_index(sample_size, train)

        # Create the raw image dataset
        raw_image_dataset = self._generate_raw_image(image_index, train)

        # Create the covariate in image
        covariate_image = self._generate_covariate_image(sample_size, image_index, train)

        # Generate the treatment D_i
        treatment = self._generate_treatment(covariate, covariate_image)

        # Generate the post-treatment covariate P_i^V
        post_treatment = self._generate_post_treatment(covariate, treatment)

        # Create the post-treated image dataset
        post_treatment_image_dataset = self._generate_image_post_treatment(
            raw_image_dataset, post_treatment
        )

        # Generate the outcome Y_i
        outcome = self._generate_outcome(covariate, treatment, covariate_image, post_treatment)

        created_dataset = {
            "train": train,
            "covariate": covariate,
            "image_index": image_index,
            "raw_image_dataset": raw_image_dataset,
            "covariate_image": covariate_image,
            "treatment": treatment,
            "post_treatment": post_treatment,
            "post_treatment_image_dataset": post_treatment_image_dataset,
            "outcome": outcome,
        }

        return created_dataset
