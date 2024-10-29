# tests/test_plot_utils.py
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch

from deepglue.plot_utils import plot_random_sample
from deepglue.plot_utils import plot_random_category_sample
from deepglue.plot_utils import convert_for_plotting  
from deepglue.plot_utils import visualize_prediction


@pytest.fixture
def create_populated_test_split_dirs(tmp_path):
    """
    Creates temporary train, valid, and test directories with sample images 
    across multiple categories, ensuring valid RGB images for testing.

    Directory structure:
        tmp_path/
            train/
                class0/   [3 images]
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
    tmp_path : Path
        The path to the temporary data directory with populated splits.
    """
    # Create the train, valid, and test directories with class0 and class1
    split_types = ['train', 'valid', 'test']
    category_image_counts = {
        'class0': [3, 1, 0],  # Train: 3, Valid: 1, Test: 0
        'class1': [2, 4, 3]   # Train: 2, Valid: 4, Test: 3
    }

    # Iterate over categories first
    for category, counts in category_image_counts.items():
        # Create the category directories for each split type
        for split, count in zip(split_types, counts):
            split_dir = tmp_path / split
            split_dir.mkdir(exist_ok=True)  # Ensure the split dir exists

            category_dir = split_dir / category
            category_dir.mkdir()

            # Create the specified number of dummy images
            for img_index in range(count):
                img_path = category_dir / f'image_{img_index}.png'
                # Generate an RGB image with varying red channel value
                image = Image.new('RGB', (32, 32), (img_index * 40 % 256, 0, 0))
                image.save(img_path)

    return tmp_path


def test_plot_random_sample(create_populated_test_split_dirs):
    """Test that plot_sample() runs without error and returns valid objects."""
    data_path = create_populated_test_split_dirs  # Use the temporary test directory fixture
    category_map = {'class0': 'category0', 'class1': 'category1'}

    # Call the function and check the returned objects
    fig, axes = plot_random_sample(data_path, category_map, split_type='train', num_to_plot=4)

    # Assertions to ensure valid matplotlib objects are returned
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure object."
    assert isinstance(axes, np.ndarray), "Expected an ndarray of Axes objects."
    assert all(isinstance(ax, plt.Axes) for ax in axes.flat), "All elements should be Axes."

    plt.close(fig)  # Close the plot to avoid memory leaks


def test_plot_random_category_sample(create_populated_test_split_dirs):
    """Test that plot_category_sample() runs without error and returns valid objects."""
    data_path = create_populated_test_split_dirs  # Use the temporary test directory fixture

    # Call the function with one category
    fig, axes = plot_random_category_sample(data_path, category='class0', split_type='train', num_to_plot=2)

    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure object."
    assert isinstance(axes, np.ndarray), "Expected an ndarray of Axes objects."
    assert all(isinstance(ax, plt.Axes) for ax in axes.flat), "All elements should be Axes."
    expected_title = "Category: class0 (train split)"
    assert fig._suptitle.get_text() == expected_title, "Plot title does not match the expected category."

    plt.close(fig)  # Close the plot to avoid memory leaks


@pytest.mark.parametrize("channels, height, width", [(1, 28, 28), (3, 32, 32)])
def test_convert_for_plotting(channels, height, width):
    """
    Test that convert_for_plotting correctly converts both grayscale and RGB tensors.
    """
    tensor = torch.rand((channels, height, width), dtype=torch.float32)
    converted_tensor = convert_for_plotting(tensor)

    assert isinstance(converted_tensor, torch.Tensor)
    assert converted_tensor.dtype == torch.uint8
    assert converted_tensor.shape == (height, width, 3)
    assert converted_tensor.min().item() >= 0
    assert converted_tensor.max().item() <= 255


def test_convert_for_plotting_non_tensor():
    """
    Test that convert_for_plotting raises a ValueError when input is not a tensor.
    """
    with pytest.raises(ValueError, match="Expected input to be a torch.Tensor"):
        convert_for_plotting([[1, 2], [3, 4]])  # Passing a non-tensor input


def test_convert_for_plotting_invalid_shape():
    """
    Test that convert_for_plotting raises a ValueError for invalid input shapes.
    """
    invalid_tensor = torch.rand((5, 28, 28))  # Invalid: 5 channels instead of 1 or 3
    with pytest.raises(ValueError, match="Expected tensor shape"):
        convert_for_plotting(invalid_tensor)


def test_visualize_prediction(caplog):
    """
    Test the visualize_prediction function.

    Verify that a warning is logged when the top_n parameter exceeds the number of available categories,
    the function returns exactly two objects, each returned object is an instance of matplotlib.axes.Axes.

    Uses caplog fixture to capture log messages emitted during the test.
    """
    # Mock input tensor (3-channel RGB image of 32x32)
    tensor = torch.rand(1, 3, 32, 32)  # Shape (1, 3, H, W)

    # Mock probabilities (5 classes)
    probabilities = torch.tensor([0.1, 0.05, 0.6, 0.05, 0.2]).unsqueeze(0)  # Shape (1, 5)

    # Mock category map for 5 categories
    category_map = {'0': 'cat', '1': 'dog', '2': 'bird', '3': 'car', '4': 'train'}

    # Capture logging output during the function call
    with caplog.at_level(logging.WARNING):
        fig, axes = visualize_prediction(tensor, probabilities, category_map, top_n=10, logscale=True)

    # Check that the expected warning was logged
    assert "top_n (10) is greater than the number of categories" in caplog.text, \
        "Expected warning message not found in logs."
    assert len(axes) == 2, "Expected two axes to be returned."
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure object."
    assert isinstance(axes, np.ndarray), "Expected an ndarray of Axes objects."
    assert all(isinstance(ax, plt.Axes) for ax in axes.flat), "All elements should be Axes."

    plt.close()  # Close the plot to avoid memory leaks
