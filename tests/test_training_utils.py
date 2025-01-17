# tests/test_training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import pytest

from deepglue.training_utils import train_one_epoch  
from deepglue.training_utils import validate_one_epoch
from deepglue.training_utils import train_and_validate
from deepglue.training_utils import accuracy
from deepglue.training_utils import predict_all
from deepglue.training_utils import predict_batch
from deepglue.training_utils import prepare_ordered_data
from deepglue.training_utils import extract_features

from conftest import BATCH_SIZE, NUM_CLASSES, NUM_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FEATURE_MAPS
from conftest import simple_transform

def test_accuracy():
    # Example 1: Simple case with batch size 4 and 3 classes
    outputs = torch.tensor([
        [0.2, 0.5, 0.3],  # Image 1: Class 1 has highest score
        [0.1, 0.3, 0.6],  # Image 2: Class 2 has highest score
        [0.8, 0.1, 0.1],  # Image 3: Class 0 has highest score
        [0.4, 0.4, 0.2]   # Image 4: Tie between Class 0 and 1, picks 0 by default
    ])
    targets = torch.tensor([1, 2, 0, 0])  # Ground truth labels

    # Test for top-1 accuracy
    top1_accuracy = accuracy(outputs, targets, topk=(1,))
    assert top1_accuracy == [100.0], f"Expected [100.0] but got {top1_accuracy}"

    # Test for top-1 and top-2 accuracy
    top2_accuracy = accuracy(outputs, targets, topk=(1, 2))
    assert top2_accuracy == [100.0, 100.0], f"Expected [100.0, 100.0] but got {top2_accuracy}"

    # Example 2: A case with some incorrect predictions
    outputs2 = torch.tensor([
        [0.6, 0.2, 0.2],  # Image 1: Incorrect (should be 1)
        [0.1, 0.8, 0.1],  # Image 2: Correct (1)
        [0.1, 0.3, 0.6],  # Image 3: Correct (2)
        [0.7, 0.2, 0.1]   # Image 4: Incorrect (should be 1)
    ])
    targets2 = torch.tensor([1, 1, 2, 1])

    # Test for top-1 and top-2 accuracy (should be 50% and 100%)
    top2_accuracy = accuracy(outputs2, targets2, topk=(1, 2))
    assert top2_accuracy == [50.0, 100.0], f"Expected [50.0, 100.0] but got {top2_accuracy}"

    
def test_train_one_epoch(simple_cnn_model, dummy_image_data):
    """
    Test for the train_one_epoch function.
    
    This test verifies that the train_one_epoch function runs without errors, does some basic
    checking on the training, and returns values of the expected types and shapes. 
    
    It uses the simple_cnn_model and dummy_image_data fixtures.

    Parameters
    ----------
    simple_cnn_model : SimpleCNNModel
        An instance of the basic CNN model used for testing.
    dummy_image_data : tuple
        The dummy dataset fixture providing input images and labels.
    """
    # Unpack the dummy data
    images, labels = dummy_image_data
    model = simple_cnn_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE) # small batch size for testing purposes

    # Set up the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    device = 'cpu'

    # Run train_one_epoch
    epoch_loss, epoch_topk_acc = train_one_epoch(model, 
                                                 dataloader, 
                                                 loss_function, 
                                                 optimizer, 
                                                 device,
                                                 topk=(1, 2))

    # Assertions
    assert isinstance(epoch_loss, float), "Epoch loss should be a float."
    assert isinstance(epoch_topk_acc, np.ndarray), "Epoch top-k accuracy should be a numpy array."
    assert epoch_topk_acc.shape[0] == 2, f"Top-k accuracy should have length 2, got {epoch_topk_acc.shape[0]}."
    assert epoch_loss >= 0, "Loss should be non-negative."
    assert epoch_topk_acc[0] <= epoch_topk_acc[1], "Top-k accuracy values should be non-decreasing."


def test_validate_one_epoch(simple_cnn_model, dummy_image_data):
    """
    Test the validate_one_epoch function.
    
    This test verifies that the validate_one_epoch function runs without errors, processes
    the data correctly, and returns values of the expected types and shapes.

    It uses the simple_cnn_model and dummy_image_data fixtures.

    Parameters
    ----------
    simple_cnn_model : SimpleCNNModel
        An instance of the basic CNN model used for testing.
    dummy_image_data : tuple
        The dummy dataset fixture providing input images and labels.
    """
    # Unpack the dummy data
    images, labels = dummy_image_data
    model = simple_cnn_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE) # small batch size for testing purposes

    # Set up the loss function
    loss_function = nn.CrossEntropyLoss()
    device = 'cpu'

    # Run validate_one_epoch
    epoch_loss, epoch_topk_acc = validate_one_epoch(model, 
                                                    dataloader, 
                                                    loss_function, 
                                                    device,
                                                    topk=(1, 2))

    # Assertions
    assert isinstance(epoch_loss, float), "Epoch loss should be a float."
    assert isinstance(epoch_topk_acc, np.ndarray), "Epoch top-k accuracy should be a numpy array."
    assert epoch_topk_acc.shape[0] == 2, f"Top-k accuracy should have length 2, got {epoch_topk_acc.shape[0]}."
    assert epoch_loss >= 0, "Loss should be non-negative."
    assert epoch_topk_acc[0] <= epoch_topk_acc[1], "Top-k accuracy values should be non-decreasing."


def test_train_and_validate(simple_cnn_model, dummy_image_data):
    """
    Test for the train_and_validate function.
    
    This test verifies that the train_and_validate function runs for two epochs without errors,
    processes the data correctly (training and validation), and returns values of the expected types and shapes.

    It uses the simple_cnn_model and dummy_image_data fixtures.

    Parameters
    ----------
    simple_cnn_model : SimpleCNNModel
        An instance of the basic CNN model used for testing.
    dummy_image_data : tuple
        The dummy dataset fixture providing input images and labels.
    """
    # Unpack the dummy data
    images, labels = dummy_image_data
    model = simple_cnn_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE) # small batch size for testing purposes

    # Set up the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Run train_and_validate
    model, history = train_and_validate(model, 
                                        train_data_loader=dataloader, 
                                        valid_data_loader=dataloader,  # Using the same data for simplicity
                                        loss_function=loss_function, 
                                        optimizer=optimizer, 
                                        device='cpu',
                                        topk=(1, 2),
                                        epochs=2)  # Run only 2 epochs for testing speed

    # Assertions for the model and history output
    assert isinstance(history, dict), "History should be a dictionary."
    assert 'train_loss' in history and 'val_loss' in history, "History should contain 'train_loss' and 'val_loss' keys."
    assert 'train_topk_accuracy' in history and 'val_topk_accuracy' in history, "History should contain accuracy keys."

    # Assertions for the loss values in the history
    assert len(history['train_loss']) == 2, "Length of 'train_loss' should match number of epochs."
    assert len(history['val_loss']) == 2, "Length of 'val_loss' should match number of epochs."
    assert all(isinstance(val, float) for val in history['train_loss']), "Train loss values should be floats."
    assert all(isinstance(val, float) for val in history['val_loss']), "Validation loss values should be floats."

    # Assertions for the accuracy values in the history
    assert len(history['train_topk_accuracy']) == 2, "Length of 'train_topk_accuracy' should match number of epochs."
    assert len(history['val_topk_accuracy']) == 2, "Length of 'val_topk_accuracy' should match number of epochs."
    assert all(isinstance(val, np.ndarray) for val in history['train_topk_accuracy']), "Train accuracies should be numpy arrays."
    assert all(isinstance(val, np.ndarray) for val in history['val_topk_accuracy']), "Validation accuracies should be numpy arrays."

    # Check that the top-1 accuracy is always less than or equal to top-2 accuracy in each epoch (in training and validation)
    for epoch_topk_acc in history['train_topk_accuracy']:
        assert epoch_topk_acc[0] <= epoch_topk_acc[1], "Train top-k accuracy values should be non-decreasing."
    for epoch_topk_acc in history['val_topk_accuracy']:
        assert epoch_topk_acc[0] <= epoch_topk_acc[1], "Validation top-k accuracy values should be non-decreasing."


def test_predict_all(simple_cnn_model, dummy_image_data):
    """
    Test the predict_all function to ensure it returns the expected output shapes and types.
    
    This test checks that:
    - all_predictions and all_labels have the same number of samples as provided by the data loader.
    - all_probabilities has the shape (num_samples, num_classes) and contains valid probabilities.
    """

    # Unpack the dummy data
    images, labels = dummy_image_data
    model = simple_cnn_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE) # small batch size for testing purposes

    # Act
    all_predictions, all_labels, all_probabilities = predict_all(model, dataloader, device='cpu')

    # Assert
    assert isinstance(all_predictions, torch.Tensor), "all_preds should be a torch.Tensor"
    assert isinstance(all_labels, torch.Tensor), "all_labels should be a torch.Tensor"
    assert isinstance(all_probabilities, torch.Tensor), "probability_matrix should be a torch.Tensor"
    
    assert all_predictions.shape == (NUM_SAMPLES,), f"Expected shape {(NUM_SAMPLES,)} for all_predictions, got {all_predictions.shape}"
    assert all_labels.shape == (NUM_SAMPLES,), f"Expected shape {(NUM_SAMPLES,)} for all_labels, got {all_labels.shape}"
    assert all_probabilities.shape == (NUM_SAMPLES, NUM_CLASSES), \
        f"Expected shape {(NUM_SAMPLES, NUM_CLASSES)} for all_probabilities, got {all_probabilities.shape}"
    
    # Check if probabilities are valid (between 0 and 1)
    assert torch.all((all_probabilities >= 0) & (all_probabilities <= 1)), "Probabilities should be between 0 and 1"
    assert torch.allclose(all_probabilities.sum(dim=1), torch.ones_like(all_probabilities.sum(dim=1))), \
        "Each row in all_probabilities should sum to 1"
    

def test_predict_batch(simple_cnn_model, dummy_image_data):
    """
    Test the predict_batch function.

    This test verifies that the predict_batch function runs without errors, returns a tensor of 
    predicted probabilities, and outputs a tensor with the expected shape and valid probability values

    Parameters
    ----------
    simple_cnn_model : nn.Module
        A simple CNN model created using the fixture for testing.
    dummy_image_data : tuple
        The dummy dataset fixture providing input images and labels.
    """
    # Unpack the dummy data
    images, _ = dummy_image_data  # Only need images for prediction
    model = simple_cnn_model
    batch_size, num_classes = images.shape[0], NUM_CLASSES

    # Run the predict_batch function
    probabilities = predict_batch(model, images, device='cpu')

    # Assertions 
    assert isinstance(probabilities, torch.Tensor), "Output should be a torch.Tensor."
    assert probabilities.shape == (batch_size, num_classes), \
        f"Expected shape {(batch_size, num_classes)}, but got {probabilities.shape}."
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), \
        "All probabilities should be in the range [0, 1]."
    assert torch.allclose(probabilities.sum(dim=1), torch.tensor(1.0)), \
        "Each row of probabilities should sum to 1."
    

def test_prepare_ordered_data(setup_test_dataset):
    """
    Test the prepare_ordered_data function to ensure it correctly prepares
    the image paths and data loader.  
    """
    data_path = setup_test_dataset
    image_paths, ordered_loader = prepare_ordered_data(data_path=data_path,
                                                       transform=simple_transform,
                                                       num_workers=0,
                                                       batch_size=2,
                                                       split_type='valid')

    # Total number of paths should equal number of validation images
    expected_image_count = 5  # Total images in 'valid' split (class0: 1, class1: 4)
    assert len(image_paths) == expected_image_count, "Image paths count mismatch."

    # Check that the DataLoader loads the correct number of images
    loaded_image_count = 0
    for batch_images, _ in ordered_loader:
        loaded_image_count += len(batch_images)
    assert loaded_image_count == expected_image_count, "Mismatch in total images returned by DataLoader."


def test_extract_features(setup_test_dataset, simple_cnn_model):
    """
    Test the extract_features function using an ordered
    DataLoader created using prepare_ordered_data.
    """
    # Use the setup_test_dataset fixture to create a temporary dataset
    data_path = setup_test_dataset

    # Create the ordered DataLoader and image paths using prepare_ordered_data
    image_paths, ordered_loader = prepare_ordered_data(data_path=data_path,
                                                       transform=simple_transform,
                                                       num_workers=0,
                                                       batch_size=2,
                                                       split_type="valid")

    # Use the simple CNN model as the base for feature extraction
    feature_extractor = create_feature_extractor(simple_cnn_model, return_nodes={"conv": "features"})
    layer_name = "features"

    features, extracted_labels = extract_features(ordered_loader, feature_extractor, layer_name, device="cpu")

    # Check the feature shape
    expected_feature_size = NUM_FEATURE_MAPS * IMAGE_HEIGHT * IMAGE_WIDTH   # flattened feature size
    expected_image_count = len(image_paths)
    assert features.shape == (expected_image_count, expected_feature_size), "Feature shape mismatch."

    # Check the labels shape
    assert extracted_labels.shape == (expected_image_count,), "Label shape mismatch."

    # Check that the the labels from feature extractor match those in ordered loader
    dataset_labels = [label for _, label in ordered_loader.dataset.samples]
    assert (extracted_labels == np.array(dataset_labels)).all(), "Extracted labels from feature extractor don't match data loader."

    # Test non-existent layer throws proper exception
    with pytest.raises(KeyError, match="Layer 'invalid_layer' not found in the feature extractor outputs!"):
        extract_features(ordered_loader, feature_extractor, layer="invalid_layer", device="cpu")





