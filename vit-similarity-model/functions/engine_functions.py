import time
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from math import floor
from torch.nn.utils import clip_grad_norm_
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import colormaps
import numpy as np
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seeds(seed: int = 689):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def _plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix',
                           cmap=colormaps.get_cmap(cmap='Blues')):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted', fontdict={'fontsize': 14, 'fontweight': 5})
    plt.ylabel('Actual', fontdict={'fontsize': 14, 'fontweight': 5})
    plt.show()


def _plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = []
    test_loss = []
    for i, value in enumerate(results['train_loss']):
        loss_value = results['train_loss'][i]
        test_loss_value = results['test_loss'][i]
        loss.append(loss_value)
        test_loss.append(test_loss_value)

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(loss))

    # Setup a plot
    plt.figure(figsize=(8, 8))

    # Plot loss
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))
    # Plot accuracy
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def _test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: str) -> tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """

  # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
      # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(torch.device(device)), y.to(torch.device(device))

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    time.sleep(3)
    return test_loss, test_acc


def _train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str,
                grad_clipping: bool = False,
                grad_norm_value: float = 1.5) -> tuple[float, float]:

    """Trains a PyTorch model for a single epoch, with gradient clipping enabled.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, gradient clipping, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        grad_clipping: If grad clipping is desired while training, default is False.
        grad_norm_value: Normalization value for grad clipping, default is 1.0.

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.1112, 0.8743)
    """

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (x, y) in enumerate(dataloader):
        # Send data to target device
        x, y = torch.tensor(x).to(torch.device(device)), y.to(torch.device(device))

        # 1. Forward pass
        y_pred = model(x)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()
        if grad_clipping:
            max_grad_norm = grad_norm_value  # Set grad clipping value
            clip_grad_norm_(model.parameters(), max_grad_norm)

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    time.sleep(3)

    return train_loss, train_acc


def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: str,
                writer: torch.utils.tensorboard.writer.SummaryWriter,
                gradient_clipping: bool = False,
                grad_norm_value: float = 1.5) -> dict[str, list]:

    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        writer: A torch tensorboard summary writer object - can be created with the create_writer() function.
        gradient_clipping: optional = False: True or False, if gradient clipping should be enabled.
        grad_norm_value: optional float = 1.5: The normalization value for gradient clipping.
    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
    """

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # put the model on the correct device
    model.to(torch.device(device))

    train_step = _train_step

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        time.sleep(2)
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           grad_clipping=gradient_clipping,
                                           grad_norm_value=grad_norm_value)

        test_loss, test_acc = _test_step(model=model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn,
                                         device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc*100:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc*100:.4f}"
        )
        time.sleep(2)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc},
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass

        # Plotting confusion matrix on the last epoch
        if epoch == epochs - 1:
            model.eval()

            true_labels = []
            pred_labels = []

            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_dataloader):

                    X, y = X.to(torch.device(device)), y.to(torch.device(device))

                    # 1. Forward pass
                    test_pred_logits = model(X)

                    #2. Argmax of results
                    test_pred_labels = test_pred_logits.argmax(dim=1)

                    #3. Sending to CPU
                    test_pred_labels = np.array(test_pred_labels.cpu())
                    y = np.array(y.cpu())

                    true_labels.extend(y)
                    pred_labels.extend(test_pred_labels)
                    #4. Plotting confusion matrix
            _plot_confusion_matrix(y_pred=pred_labels, y_true=true_labels, classes=
                                   test_dataloader.dataset.dataset.classes)
            _plot_loss_curves(results=results)
    return results


def organize_datasets(train_test_split: float = .75,
                      data_transforms: transforms = transforms.Compose(
                          [

                          ]
                      )) -> tuple[DataLoader, DataLoader]:
    """
    This function organizes the page-category-classification-model dataset and returns it.

    as a dataloader object to be used for training/testing.

    Returns:
        train_dataloader (Dataloader): A dataloader object to use for training.
        test_dataloader (Dataloader): A dataloader object to use for testing.
    """
    # Setting up Dataloader from Data
    if transforms:
        train_transform_composed = data_transforms
    else:
        train_transform_composed = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    # getting image path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir_path = os.path.join(current_file_dir, "..", "data")
    full_dataset = datasets.ImageFolder(
        root=image_dir_path, transform=train_transform_composed
    )

    # splitting raw dataset
    train_size = int(floor(train_test_split * len(full_dataset)))
    test_size = int(floor(len(full_dataset) - train_size))
    train_data, test_data = data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=16)

    return train_dataloader, test_dataloader
