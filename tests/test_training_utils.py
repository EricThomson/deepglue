# tests/test_training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytest

from deepglue import train_one_epoch  
from deepglue import validate_one_epoch
from deepglue import accuracy


# Constants used to build dummy data/networks in fixtures across these tests
INPUT_SIZE = 10
NUM_CLASSES = 3
NUM_SAMPLES = 10

@pytest.fixture
def simple_linear_model(scope="module"):
    """
    Fixture for creating a simple neural network model.
    
    This fixture defines and returns an instance of a basic feedforward neural network
    with a single linear layer, using INPUT_SIZE inputs and NUM_CLASSES outputs.

    Returns
    -------
    SimpleLinearModel
        An instantiated neural network model with one linear layer.
    """
    class SimpleLinearModel(nn.Module):
        def __init__(self):
            super(SimpleLinearModel, self).__init__()
            self.fc = nn.Linear(INPUT_SIZE, NUM_CLASSES)
        
        def forward(self, x):
            return self.fc(x)
    
    return SimpleLinearModel()


@pytest.fixture
def dummy_data(scope="module"):
    """
    Fixture for generating dummy data for testing.
    
    Creates NUM_SAMPLES for network with INPUT_SIZE features and NUM_CLASSES outputs. 

    Returns
    -------
    tuple
        A tuple (test_features, test_labels) where:
        - test_features (torch.Tensor): Randomly generated input features with shape (NUM_SAMPLES, INPUT_SIZE).
        - test_labels (torch.Tensor): Randomly generated integer labels (0...NUM_CLASSES) corresponding to the classes.
    """
    test_features = torch.randn(NUM_SAMPLES, INPUT_SIZE)
    test_labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return test_features, test_labels


def test_train_one_epoch(simple_linear_model, dummy_data):
    """
    Test for the train_one_epoch function.
    
    This test verifies that the train_one_epoch function runs without errors, does some basic
    checking on the training, and returns values of the expected types and shapes. It uses
    the simple_linear_model and dummy_data fixtures to set up the required components.

    Parameters
    ----------
    simple_linear_model : SimpleLinearModel
        An instance of the basic neural network model used for testing.
    dummy_data : tuple
        The dummy dataset fixture providing input features and labels.
    """
    # Unpack the dummy data
    test_features, test_labels = dummy_data

    # Use the instantiated model from the fixture directly
    model = simple_linear_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(test_features, test_labels)
    dataloader = DataLoader(dataset, batch_size=2) # small batch size for testing purposes

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


def test_validate_one_epoch(simple_linear_model, dummy_data):
    """
    Test the validate_one_epoch function.
    
    This test verifies that the validate_one_epoch function runs without errors, processes
    the data correctly, and returns values of the expected types and shapes.

    Parameters
    ----------
    simple_linear_model : SimpleLinearModel
        An instance of the basic neural network model used for testing.
    dummy_data : tuple
        The dummy dataset fixture providing input features and labels.
    """
    # Unpack the dummy data
    test_features, test_labels = dummy_data

    # Use the instantiated model from the fixture directly
    model = simple_linear_model

    # Create a DataLoader using the dummy data
    dataset = TensorDataset(test_features, test_labels)
    dataloader = DataLoader(dataset, batch_size=2)  # small batch size for testing purposes

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


def test_accuracy():
    # Example 1: Simple case with batch size 4 and 3 classes
    output = torch.tensor([
        [0.2, 0.5, 0.3],  # Image 1: Class 1 has highest score
        [0.1, 0.3, 0.6],  # Image 2: Class 2 has highest score
        [0.8, 0.1, 0.1],  # Image 3: Class 0 has highest score
        [0.4, 0.4, 0.2]   # Image 4: Tie between Class 0 and 1, picks 0 by default
    ])
    target = torch.tensor([1, 2, 0, 0])  # Ground truth labels

    # Test for top-1 accuracy
    top1_accuracy = accuracy(output, target, topk=(1,))
    assert top1_accuracy == [100.0], f"Expected [100.0] but got {top1_accuracy}"

    # Test for top-1 and top-2 accuracy
    top2_accuracy = accuracy(output, target, topk=(1, 2))
    assert top2_accuracy == [100.0, 100.0], f"Expected [100.0, 100.0] but got {top2_accuracy}"

    # Example 2: Another controlled case to check top-k behavior
    output = torch.tensor([
        [0.2, 0.7, 0.1],  # Image 1: Class 1 is correct, but Class 2 is in top-2
        [0.9, 0.05, 0.05],# Image 2: Class 0 correct, Class 1 also in top-2
        [0.1, 0.2, 0.7],  # Image 3: Class 2 correct
        [0.3, 0.6, 0.1]   # Image 4: Class 1 correct
    ])
    target = torch.tensor([1, 0, 2, 1])

    # Test for top-1 accuracy (should be 100%)
    top1_accuracy = accuracy(output, target, topk=(1,))
    assert top1_accuracy == [100.0], f"Expected [100.0] but got {top1_accuracy}"

    # Test for top-1 and top-2 accuracy (should be 100% and 100%)
    top2_accuracy = accuracy(output, target, topk=(1, 2))
    assert top2_accuracy == [100.0, 100.0], f"Expected [100.0, 100.0] but got {top2_accuracy}"

    # Example 3: A case with some incorrect predictions
    output = torch.tensor([
        [0.6, 0.2, 0.2],  # Image 1: Incorrect (should be 1)
        [0.1, 0.8, 0.1],  # Image 2: Correct (1)
        [0.1, 0.3, 0.6],  # Image 3: Correct (2)
        [0.7, 0.2, 0.1]   # Image 4: Incorrect (should be 1)
    ])
    target = torch.tensor([1, 1, 2, 1])

    # Test for top-1 and top-2 accuracy (should be 50% and 100%)
    top2_accuracy = accuracy(output, target, topk=(1, 2))
    assert top2_accuracy == [50.0, 100.0], f"Expected [50.0, 100.0] but got {top2_accuracy}"