import matplotlib.pyplot as plt


def visualize_dataset(dataset: dict, max_size: int = 10):
    """
    Visualize the dataset.
    """
    n = min(max_size, len(dataset["post_treatment_image_dataset"]))
    _fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(5, 2 * n))

    for i in range(n):
        treated_img = dataset["post_treatment_image_dataset"][i][0].squeeze(0)

        # Plot original image
        axes[i, 0].imshow(dataset["raw_image_dataset"][i][0], cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Original {i}")

        # Plot post-treatment image
        axes[i, 1].imshow(treated_img, cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Post-treatment {i}")

    plt.tight_layout()
    plt.show()
