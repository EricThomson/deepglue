# tests/test_training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytest

from deepglue import train_one_epoch  # Adjust import based on your module structure
from deepglue import accuracy


# Constants used to build dummy data/networks in fixtures across these tests
INPUT_SIZE = 10
NUM_CLASSES = 3
NUM_SAMPLES = 10

@pytest.fixture
def simple_linear_model():
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
def dummy_data():
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