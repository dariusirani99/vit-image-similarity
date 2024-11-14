from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from srcs.model_architecture import PreTrainedViT
import os
import yaml
import torch
from torch.utils.data import DataLoader
from training.train_step import train_step
from training.test_step import test_step
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colormaps
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

# Labels Variable
labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
          'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup',
          'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
          'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
          'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
          'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


def plot_loss_curves(
    results: dict[str, list[float]], logging_folder: str = "model_logging"
):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        logging_folder (str, default='model_logging'): The folder in the repository to record logging.
    """

    # Get the loss values of the results dictionary (training and test)
    loss = []
    test_loss = []
    for i, value in enumerate(results["train_loss"]):
        loss_value = results["train_loss"][i]
        test_loss_value = results["test_loss"][i]
        loss.append(loss_value)
        test_loss.append(test_loss_value)

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(loss))

    # Setup a plot
    plt.figure(figsize=(8, 8))

    # model logging folder save paths
    current_file_dir = Path(__file__).resolve().parent
    logging_path = (current_file_dir / ".." / logging_folder).resolve()
    save_path_losses = logging_path / "model_losses.png"
    save_path_acc = logging_path / "model_accuracy.png"

    # Plot loss
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(save_path_losses)
    plt.close()

    plt.figure(figsize=(8, 8))
    # Plot accuracy
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(save_path_acc)
    plt.close()


def plot_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    classes: list,
    logging_folder: str = "model_logging",
    title="Confusion Matrix",
    cmap=colormaps.get_cmap(cmap="Blues"),
):
    """
    Plots the confusion matrix for a specific model true labels and predicted labels tensor.

    Args:
        y_true (np.array): The true labels, converted to an np.array
        y_pred (np.array): The model predicted labels, converted to an np.array
        classes (List): List of classes, in alphabetical order
        logging_folder (str, default='model_logging'): The path in the repository to record logging.
        title: Plot title as a string
        cmap (colormaps object): Type of coloring to use for confusion matrix
    Returns:
        A saved plot of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.xlabel("Predicted", fontdict={"fontsize": 14, "fontweight": 5})
    plt.ylabel("Actual", fontdict={"fontsize": 14, "fontweight": 5})
    current_file_dir = Path(__file__).resolve().parent
    logging_path = (current_file_dir / ".." / logging_folder).resolve()
    save_path = logging_path / "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()


def prepare_datasets(transform: torchvision.transforms, batch_size: int):
    trainset = CIFAR100(root='./training/data', train=True, download=True, transform=transform)
    testset = CIFAR100(root='./training/data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size)
    return train_loader, test_loader


def load_config(config_name: str):
    """Loads the config file."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "config")
    config_file_path = os.path.join(config_path, f"{config_name}")
    with open(config_file_path) as file:
        return yaml.safe_load(file)


def set_seeds(seed: int = 689):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 689.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def train_model():
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train() and test()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    :returns: A saved .pt file, along with loss curves, saved in appropriate directories.
    """
    torch.cuda.empty_cache()

    # getting config variables
    config = load_config(config_name="train_config.yml")
    model_name = config["model_name"]
    seeds = config["training_seeds"]
    batch_size = config["data_batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    gradient_clipping_norm_value = config["gradient_clipping_norm_value"]

    # getting datasets and dataloaders
    transform_composed = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["preprocessing"]["mean"],
                std=config["preprocessing"]["std"],
            ),
        ]
    )

    train_loader, test_loader = prepare_datasets(transform=transform_composed, batch_size=batch_size)

    # Setting seeds for reproducibility
    set_seeds(seeds)
    # getting device
    if config["use_cuda"] or torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # loading model and making the parameters trainable
    model = PreTrainedViT(train_config=config)
    for parameter in model.parameters():
        parameter.requires_grad = True

    # defining optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # setting up results
    results: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Printing training information
    print(
        f"[INFO] Training Stats:\n       Epochs: {epochs}\n       Learning Rate: {learning_rate}\n"
    )
    print(f"[INFO] Starting training for {epochs} epochs...")
    for epoch in range(epochs):

        # calling train step for the data
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clipping=gradient_clipping,
            grad_norm_value=gradient_clipping_norm_value,
        )

        # calling test step after train step
        test_loss, test_acc = test_step(
            model=model, dataloader=test_loader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc * 100:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc * 100:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Plotting confusion matrix on the last epoch
        if epoch == epochs - 1:
            model.eval()

            true_labels = []
            pred_labels = []

            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_loader):
                    X, y = X.clone().detach().to(torch.device(device)), y.to(
                        torch.device(device)
                    )

                    # 1. Forward pass
                    test_pred_logits = model(X)

                    # 2. Argmax of results
                    test_pred_labels = test_pred_logits.argmax(dim=1)

                    # 3. Sending to CPU
                    test_pred_labels = test_pred_labels.cpu().numpy()
                    y = y.cpu().numpy()

                    true_labels.extend(y)
                    pred_labels.extend(test_pred_labels)

            # 4. Plotting confusion matrix and loss/accuracy curves'
            if plot_confusion_matrix:
                print("[INFO] Plotting confusion matrix and saving to logging path...")
                plot_confusion_matrix(
                    y_pred=pred_labels,
                    y_true=true_labels,
                    logging_folder=r"model_logging",
                    classes=labels,
                )

            if plot_loss_curves:
                print("[INFO] Plotting loss curves and saving to logging path...")
                plot_loss_curves(
                    results, logging_folder=r"model_logging"
                )

    # Saving Model .pth weights file
    model = model.to(torch.device('cpu'))
    print("[INFO] Saving model pt file to weights path...")
    torch.save(model, f"model-file/{model_name}.pth")

if __name__ == "__main__":
    train_model()
