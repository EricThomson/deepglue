"""
deepglue training_utils.py

Functions that are useful for training deep networks, including validation and testing and metrics. 
"""
import numpy as np
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm

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
    The function logs progress using the logging module. Set your loggers to 'debug' to see progress.
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
        if np.mod(batch_num, display_period) == 0:
            logging.debug(f"Starting batch {batch_num}/{num_batches}")

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        optimizer.zero_grad()  # Zero out gradients

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        if np.mod(batch_num, display_period) == 0:
            logging.debug("\tStarting backwards pass")

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
            logging.debug(f"\tLoss = {loss.item():.3f}")

    # Compute average loss and accuracy over the epoch
    epoch_loss = running_loss / total_samples
    epoch_topk_acc = [100*(correctk.item() / total_samples) for correctk in total_correct_k]

    logging.debug("Done training!")
    logging.debug(f"Training epoch loss: {epoch_loss:0.3f}")

    return epoch_loss, np.array(epoch_topk_acc)


def validate_one_epoch(model, valid_data_loader, loss_function, device, topk=(1,5)):
    """
    Validates the model for one epoch using the provided validation data loader.

    Parameters
    ----------
    model : torch model
        The neural network model to be validated.
    valid_data_loader : torch.utils.data.DataLoader
        An iterable that provides the batches for validation data set
    loss_function : callable
        The loss function to compute the loss (e.g., CrossEntropyLoss).
    device : str
        The device ('cpu' or 'cuda') on which the model and data are placed.
    topk: tuple of ints
        A tuple specifying which top-k accuracies to calculate. Defaults to (1,5)

    Returns
    -------
    epoch_loss : float
        The average loss over all samples in the validation epoch.
    epoch_topk_acc : list of floats
        A list of average top-k accuracies over all samples in the epoch. 

    Notes
    -----
    Runs in evaluation mode (`model.eval()`) and gradient calculations are disabled.
    """

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # initialize 
    running_loss = 0.0
    total_correct_k = [0.0] * len(topk)  # To accumulate the total number of correct predictions at each k level
    total_samples = 0

    num_batches = len(valid_data_loader)
    display_period = max(5, int(0.05*num_batches))
    logging.debug(f"Starting validation on {num_batches} batches.")
    logging.debug(f"Display period {display_period}")

    with torch.no_grad():  # Disable gradient calculation for validation
        # data loader will cycle through all batches in one epoch
        for batch_num, (inputs, labels) in enumerate(valid_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_size = inputs.size(0)
            total_samples += batch_size

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Loss
            running_loss += loss.item() * batch_size

            # Calculate number correct for each k using the accuracy function
            batch_k_accuracies = accuracy(outputs, labels, topk=topk)
            for i, acc_k in enumerate(batch_k_accuracies):
                num_correct_k = acc_k * batch_size / 100
                total_correct_k[i] += num_correct_k 

            if np.mod(batch_num, display_period) == 0:
                logging.debug(f"Batch {batch_num}/{num_batches} loss = {loss.item():.3f}")

    # Compute average loss and top-k accuracies over the epoch
    epoch_loss = running_loss / total_samples
    epoch_topk_acc = [100 * (correct_k.item() / total_samples) for correct_k in total_correct_k]
    
    logging.debug("Done validation!")
    logging.debug(f"Validation epoch loss: {epoch_loss:.3f}")

    return epoch_loss, np.array(epoch_topk_acc)


def train_and_validate(model, 
                       train_data_loader, 
                       valid_data_loader, 
                       loss_function, 
                       optimizer, 
                       device, 
                       topk=(1,5),
                       epochs=25):
    """
    Train and validate a model for a given number of epochs.
    
    Parameters
    ----------
    model : torch model
        The neural network model to be trained and validated.
    train_data_loader : torch.utils.data.DataLoader
        An iterable that provides the training data batches.
    valid_data_loader : torch.utils.data.DataLoader
        An iterable that provides the batches for validation data set
    loss_function : callable
        The loss function to compute the loss (e.g., CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        The optimizer used to update model parameters during training (e.g., Adam, SGD).
    device : str
        The device ('cpu' or 'cuda') on which the model and data are placed.
    topk: tuple of ints, optional
        A tuple specifying which top-k accuracies to calculate. Defaults to (1,5)
    epochs : int, optional
        Number of epochs to run. Defaults to 25.
        
    Returns
    -------
    model : torch.nn.Module
        The trained model after the completion of training.
    history : dict
        A dictionary containing training and validation loss and top-k accuracies per epoch.
    """
    train_loss, validation_loss = [], []
    train_topk_acc, validation_topk_acc = [], []
    
    logging.info(f"Training/validation {epochs} epochs")
    
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")

        # Training step
        epoch_loss_train, epoch_train_topk = train_one_epoch(model, 
                                                             train_data_loader, 
                                                             loss_function, 
                                                             optimizer, 
                                                             device,
                                                             topk=topk)
        logging.info(f"\tTraining: Loss {epoch_loss_train:0.4f}")

        train_loss.append(epoch_loss_train)
        train_topk_acc.append(epoch_train_topk)

        # Validation step
        epoch_loss_val, epoch_val_topk = validate_one_epoch(model, 
                                                            valid_data_loader, 
                                                            loss_function, 
                                                            device,
                                                            topk=topk)

        validation_loss.append(epoch_loss_val)
        validation_topk_acc.append(epoch_val_topk)
        
        logging.info(f"\tValidation: Loss {epoch_loss_val:.4f}")
                   
        torch.cuda.empty_cache()  # Clears unused GPU memory

    logging.info("Done!")

    history = {'train_loss': np.array(train_loss), 
               'train_topk_accuracy': np.array(train_topk_acc),
               'val_loss': np.array(validation_loss),
               'val_topk_accuracy': np.array(validation_topk_acc) }               
     
    return model, history  # Return both the trained model and the history


def predict_all(model, data_loader, device='cuda'):
    """
    Make predictions for all batches of data in data loader.

    Use the model to generate predictions for all batches from the provided data loader. 
    It returns the predicted class labels, true labels, class probabilities for each sample. 

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (e.g., ResNet50).
    data_loader : torch.utils.data.DataLoader
        An iterable that provides batches of input data and their corresponding labels.
    device : str, optional
        The device ('cpu' or 'cuda') on which the model and data are placed.
        Defaults to 'cuda'.

    Returns
    -------
    all_predictions : torch.Tensor
        An array of predicted labels for each sample in the dataset, with shape (num_samples,)
    all_labels : torch.Tensor
        An array of true labels for each sample in the dataset, with shape (num_samples,)
    all_probabilities: torch.Tensor
        A 2D array of shape (num_samples, num_categories) containing the softmax-normalized
        probabilities for each category: each row represents the predicted probability 
        distribution for a single sample.
    """
    all_predictions = []
    all_labels = []
    all_probabilities = []
    num_batches = len(data_loader)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, total=num_batches, desc="Predicting Batches"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images) # logits
            _, preds = torch.max(logits, 1)
            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())

            # convert logits to probs in batch 
            probabilities = softmax(logits, dim=1)  
            all_probabilities.append(probabilities.cpu())
            
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    return all_predictions, all_labels, all_probabilities


def predict_batch(model, image_batch, device='cuda'):
    """
    Predicts the category probabilities for a batch of images

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (e.g., ResNet50).
    image_batch : torch.Tensor
        A batch of images of shape (batch_size, 3, H, W).
    device : str, optional
        The device ('cpu' or 'cuda') on which the model and data are placed.
        Defaults to cuda

    Returns
    -------
    probabilities: torch.Tensor
        Predicted probabilities for each image in the batch.
        Shape is (batch_size x num_categories)

    TODO
    ----
    - Change name to predict_sample because this isn't a batch in the conventional sense coming
    from a data loader, keep the language consistent across the package. 
    - Have it return predicted 'labels' and actual labels like predict_all does.
    """
    if device not in ['cuda', 'cpu']:
        raise ValueError(f"Invalid device: {device}. Use 'cuda' or 'cpu'.")
        
    model = model.to(device)
    image_batch = image_batch.to(device)
    
    logging.info(f"Generating predictions for {image_batch.shape[0]} samples")

    model.eval()
    with torch.no_grad():
        logits = model(image_batch)
        probabilities = softmax(logits, dim=1)
    return probabilities


def accuracy(outputs, targets, topk=(1,)):
    """
    Computes the top-k accuracy for classifier predictions.

    Calculates how often the true label is within the top-k predictions,
    for each value of k specified in `topk`. 

    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions of shape (num_samples, num_classes), where each row contains the 
        logits or probabilities for each class. 
    targets : torch.Tensor
        The ground truth labels, of shape (num_samples,) or (num_samples, num_classes) if one-hot encoded.
    topk : tuple of int, optional
        A tuple of integers specifying the values of k for which to compute the prediction accuracy.
        Defaults to (1,).

    Returns
    -------
    list of torch.Tensor
        A list of accuracy values for each specified k in `topk`, expressed as percentages.

    Notes
    -----
    - Adapted from torchvision's accuracy() function (release 0.19.1), which is licensed under the BSD-3 License.
    - Original implementation in pytorch/vision/references/classification/utils.py 
    """
    # logging.debug(f"Calculating topk accuracy with topk input {topk}")

    with torch.inference_mode():
        maxk = max(topk)
        batch_size = targets.size(0)
        # if targets are one-hot encoded
        if targets.ndim == 2:
            targets = targets.max(dim=1)[1]

        _, pred = outputs.topk(maxk, 1, True, True) # get the top k predictions
        pred = pred.t() # convert to maxk x batch_size which is what comparitor wants
        correct = pred.eq(targets.unsqueeze(0))  # k x batches bool gives position of correct prediction (if any) for batch col

        topk_accuracy = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32) # sum all correct in first k rows
            proportion_correct = correct_k/batch_size
            topk_accuracy.append(100.0*proportion_correct)

        return topk_accuracy


def extract_features(dataloader, feature_extractor, layer, device='cuda'):
    """
    Extract features from a network layer using a data loader, feature extractor, and specified layer.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset (often configured without shuffling or dropping samples)
    feature_extractor : torch.nn.Module
        The feature extractor model, note this is typically created with
        torchvision's `feature_extraction.create_feature_extractor()` built-in. 
    layer : str
        The name of the layer to extract features from. Must be present in the output of the feature extractor.
    device : str, optional
        The device to use for feature extraction ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns
    -------
    features : numpy.ndarray
        Extracted features of shape (num_images, num_flattened_features), where `num_flattened_features`
        depends on the layer output dimensions.
    labels : numpy.ndarray
        Corresponding ground-truth labels for each image, of shape (num_images,).

    Raises
    ------
    KeyError
        If the specified layer is not found in the output of the feature extractor.

    Notes
    -----
    - For large datasets, ensure sufficient memory is available for concatenating feature arrays: they can 
      grow extremely large for large network models. 

    TODO
    ----
    - Add optimizations for very large arrays (eg quantization, out-of-core computation with dask and xarray, etc).
    """

    logging.info(f"Feature extraction starting for layer '{layer}'. Setup can take a minute.")

    feature_extractor.to(device)
    
    # Initialize vars
    features = []  # To store flattened features for the specified layer
    labels = [] 

    # extract features batch by batch
    for batch_num, (batch_images, batch_labels) in tqdm(enumerate(dataloader),
                                                        desc="Extracting features",
                                                        total=len(dataloader)):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            outputs = feature_extractor(batch_images)
            
            # Get the output for the specified layer
            if layer not in outputs:
                raise KeyError(f"Layer '{layer}' not found in the feature extractor outputs!")
    
            # Flatten the features for each image in the batch
            output = outputs[layer]
            flattened_features = output.reshape(output.size(0), -1)
            features.append(flattened_features.cpu().numpy())
    
            labels.extend(batch_labels.numpy())
    
    # Concatenate features across all batches
    features = np.concatenate(features, axis=0)  # Shape: [num_images, num_flattened_features]
    labels = np.array(labels)  # Shape: [num_images]
    
    logging.info(f"Feature extraction complete for layer '{layer}'.")

    return features, labels


def prepare_ordered_data(data_path, transform, num_workers=0, batch_size=4, split_type='valid'):
    """
    Prepare ordered data loader and correponding image path list for feature extraction or 
    other pipelines that require a full dataset in order.

    Generate a list of image paths and a DataLoader for a given dataset split.
    The image path and the DataLoader indices are guaranteed to match because both `shuffle`
    and `drop_last` are set to `False`, ensuring the data will be accessed in order without
    dropping any samples.

    Parameters
    ----------
    data_path : str or Path
        Path to the root directory containing the split folders ('train', 'valid', 'test')
    transform : torchvision transform (callable)
        The transformations to apply to each image.
    num_workers : int, optional
        Number of workers for parallel data loading. Higher values improve performance
        during feature extraction but may lead to multiprocessing issues on some platforms.
        Defaults to 0 (no multiprocessing).
    batch_size : int, optional
        Batch size for the DataLoader. Larger values improve feature extraction speed
        but requires more memory. Defaults to 4.
    split_type : str, optional
        The split folder to sample from ('train', 'valid', 'test'). Defaults to 'train'.
        
    Returns
    -------
    image_paths : list of str
        A list of file paths to the images in the dataset split, in the same order as
        the DataLoader batches.
    ordered_loader : torch.utils.data.DataLoader
        A DataLoader for the ordered dataset split, configured to not shuffle data and
        to include all samples.

    Raises
    ------
    FileNotFoundError
        If the specified data paths do not exist.
        
    Notes
    -----
    - Designed for feature extraction workflows where maintaining the correspondence between 
      image file paths and DataLoader batches is critical.
    - For large datasets, consider increasing `num_workers` and `batch_size` for better performance.
    """
    data_path = Path(data_path) 
    split_path =data_path / split_type
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} does not exist.")
        
    dataset = datasets.ImageFolder(root=split_path, transform=transform)
    image_paths = [path for path,_ in dataset.samples]
    ordered_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,  # do not change
                                drop_last=False)  # do not change
    
    return image_paths, ordered_loader    