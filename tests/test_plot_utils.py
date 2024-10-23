# tests/test_plot_utils.py
import logging
import matplotlib.pyplot as plt
import pytest
import torch

from deepglue.plot_utils import convert_for_plotting  
from deepglue.plot_utils import visualize_prediction


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
        axes = visualize_prediction(tensor, probabilities, category_map, top_n=10, logscale=True)

    # Check that the expected warning was logged
    assert "top_n (10) is greater than the number of categories" in caplog.text, \
        "Expected warning message not found in logs."
    assert len(axes) == 2, "Expected two axes to be returned."
    assert isinstance(axes[0], plt.Axes), "First axis should be an instance of plt.Axes."
    assert isinstance(axes[1], plt.Axes), "Second axis should be an instance of plt.Axes."

    plt.close()  # Close the plot to avoid memory leaks
