"""
deepglue training_utils.py

Functions that are useful for training deep networks, including validation and testing and metrics. 
"""
import numpy as np
import torch

import logging
logging.getLogger(__name__)


def train_one_epoch(model, train_data_loader, loss_function, optimizer, device, topk=(1,5)):
    """
    Trains the model for one epoch using the provided training data loader.

    Parameters
    ----------
    model : torch model
        The neural network model to be trained.
    train_data_loader : torch.utils.data.DataLoader
        An iterable that provides the training data batches.
    loss_function : callable
        The loss function to compute the loss (e.g., CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        The optimizer used to update model parameters (e.g., Adam, SGD).
    device : str
        The device ('cpu' or 'cuda') on which the model and data are to be placed.
    topk: tuple of ints
        A tuple specifying which top-k accuracies to calculate. Defaults to (1,5)

    Returns
    -------
    epoch_loss : float
        The average loss over all samples in the epoch.
    epoch_topk_acc : list of floats
        A list of average top-k accuracies over all samples in the epoch. 

    Notes
    -----
    The function logs progress and epoch metrics using the logging module. Set to debug to see progress.
    """
    model.to(device)
    model.train()  # Set the model to training mode
    
    # initialize losses and sample numbers
    running_loss = 0.0
    total_correct_k = [0.0]*len(topk) # to accumulate total number correct at each k level
    total_samples = 0
    
    num_batches = len(train_data_loader)
    display_period = max(5, int(0.05*num_batches))
    logging.debug(f"Starting training on {num_batches} batches.")
    logging.debug(f"Display period {display_period}")

    # data loader will cycle through all batches in one epoch
    for batch_num, (inputs, labels) in enumerate(train_data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        optimizer.zero_grad()  # Zero out gradients

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss
        running_loss += loss.item() * batch_size 

        # Calculate number correct for each k (just using accuracy may not be good if batch sizes vary)
        batch_k_accuracies = accuracy(outputs, labels, topk=topk)
        for i, acc_k in enumerate(batch_k_accuracies):
            num_correct_k = acc_k * batch_size / 100
            total_correct_k[i] += num_correct_k 
            
        if np.mod(batch_num, display_period) == 0:
            logging.debug(f"Batch {batch_num}/{num_batches} loss = {loss.item():.3f}")

    # Compute average loss and accuracy over the epoch
    epoch_loss = running_loss / total_samples
    epoch_topk_acc = [100*(correctk.item() / total_samples) for correctk in total_correct_k]

    logging.debug("Done training!")
    logging.debug(f"Training epoch loss: {epoch_loss:0.3f}")

    return epoch_loss, np.array(epoch_topk_acc)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Parameters
    ----------
    output : torch.Tensor
        The output predictions from the model, typically of shape (batch_size, num_classes).
    target : torch.Tensor
        The ground truth labels, of shape (batch_size,) or (batch_size, num_classes) if one-hot encoded.
    topk : tuple of int, optional
        A tuple of integers specifying the values of k for which to compute the prediction accuracy.
        Defaults to (1,).

    Returns
    -------
    list of torch.Tensor
        A list of accuracy values for each specified k in `topk`, expressed as percentages.

    Notes
    -----
    - Adapted from torchvision's accuracy function (release 0.19.1), which is licensed under the BSD-3 License.
    - Original implementation in pytorch/vision/references/classification/utils.py 
    """
    # logging.debug(f"Calculating topk accuracy with topk value {topk}")

    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        # if targets are one-hot encoded
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True) # get the top k predictions
        pred = pred.t() # convert to maxk x batch_size which is what comparitor wants
        correct = pred.eq(target.unsqueeze(0))  # k x batches bool gives position of correct prediction (if any) for batch col

        topk_accuracy = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32) # sum all correct in first k rows
            proportion_correct = correct_k/batch_size
            topk_accuracy.append(100.0*proportion_correct)

        return topk_accuracy