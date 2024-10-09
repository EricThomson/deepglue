"""
deepglue training_utils.py

Functions that are useful for training deep networks, including validation and testing and metrics. 
"""

import torch

import logging
logging.getLogger(__name__)

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