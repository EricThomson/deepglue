"""
Test configuration for deep glue
Fixtures, parameters, etc for deep glue tests
"""
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
import pytest

# Constants to build dummy data/networks in fixtures across tests
NUM_SAMPLES = 10  # number of total samples for dummy dataset (10 images)
NUM_FEATURE_MAPS = 16 # number of output channels (feature maps) for simple cnn model convolution layer
BATCH_SIZE = 2 # number of batches used in test dataloaders
NUM_CLASSES = 3 # number of output classes for  classification tasks
IMAGE_HEIGHT = 32 
IMAGE_WIDTH = 32

# Some dataloaders need a transform use this simple little guy
simple_transform =   transforms.Compose([transforms.ToImage(), 
                                         transforms.ToDtype(torch.float32)])

@pytest.fixture()
def simple_cnn_model():
    """
    Fixture for creating a simple CNN model for images.

    This fixture defines and returns a minimal convolutional neural network
    with:
    - A single convolutional layer producing NUM_FEATURES feature maps.
    - A fully connected layer for classification with NUM_CLASSES outputs.
    
    This model is used in test_training_utils.
    """
    class SimpleCNNModel(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super(SimpleCNNModel, self).__init__()
            self.conv = nn.Conv2d(in_channels=3, 
                                  out_channels=NUM_FEATURE_MAPS, 
                                  kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(NUM_FEATURE_MAPS * IMAGE_HEIGHT * IMAGE_WIDTH, NUM_CLASSES)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)  # Flatten
            return self.fc(x)
    
    return SimpleCNNModel()


@pytest.fixture()
def dummy_image_data():
    """
    Fixture for generating a batch of dummy image data.

    Creates random image data with shape (NUM_SAMPLES, 3, IMAGE_HEIGHT, IMAGE_WIDTH) 
    and random labels for each image.

    Returns
    -------
    images (torch.Tensor): Randomly generated images
    labels (torch.Tensor): Randomly generated integer labels (len 10)
    """
    images = torch.randn(NUM_SAMPLES, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return images, labels


@pytest.fixture
def setup_test_dataset(tmp_path):
    """
    Sets up temporary test dataset for train, valid, and test with sample images
    in two categories.

    Standard structure:
        tmp_path/
            train/
                class0/   [3 images] image_0.png, image_1.png, image_2.png
                class1/   [2 images]
            valid/
                class0/   [1 image]
                class1/   [4 images]
            test/
                class0/   [0 images]
                class1/   [3 images]

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory unique to each test.

    Returns
    -------
    tmp_path: Path
        The path to the temporary data directory.
    """
    # Define the categories and the number of images per split
    split_types = ['train', 'valid', 'test']
    category_image_counts = {
        'class0': [3, 1, 0],  # Train: 3, Valid: 1, Test: 0
        'class1': [2, 4, 3]   # Train: 2, Valid: 4, Test: 3
    }

    # Iterate through categories first, then through the splits
    for category, image_counts in category_image_counts.items():
        for split, count in zip(split_types, image_counts):
            # Create the split and category directories
            split_dir = tmp_path / split
            category_dir = split_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Create the specified number of dummy image files
            for i in range(count):
                image_path = category_dir / f'image_{i}.png'
                # Create a simple solid-color image
                img = Image.new('RGB', 
                                (IMAGE_WIDTH, IMAGE_HEIGHT), 
                                color=(i * 20 % 255, i * 30 % 255, i * 40 % 255))
                img.save(image_path)

    return tmp_path