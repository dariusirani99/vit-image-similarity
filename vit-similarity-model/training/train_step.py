import warnings

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clipping: bool = False,
    grad_norm_value: float = 1.0,
) -> tuple:
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
    train_loss, train_acc = 0.0, 0.0

    # Loop through data loader data batches
    for batch, (x, y) in enumerate(dataloader):
        # Send data to target device
        x, y = torch.tensor(x).to(device), y.to(device)

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
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc
