"""
deepglue plot_utils.py

Module includes functions that are useful for plotting/visualization during different
deep learning tasks
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from skimage.color import rgb2gray
import torch

import logging
logging.getLogger(__name__)


# Set global style for all plots when the module is imported
plt.rcParams.update({'figure.titlesize': 16, # suptitle 
                     'axes.titlesize': 14, # individual plot titles
                    'axes.labelsize': 12,  # x and y labels
                    'legend.fontsize': 10.5, # legend labels 
                    'xtick.labelsize': 10, # x- and y-tick labels are smallest
                    'ytick.labelsize': 10})  # suptitle


def plot_category_samples(data_path, category, split_type='train', num_to_plot=16, filetype='png'):
    """
    Plots random samples from a specified category within a data split (default is train split).

    Assumes a directory structure where images are stored in category-specific 
    subdirectories in parent data dir that contains 'train', 'valid', and 'test' folders.

    Parameters
    ----------
    data_path : Path
        The path to the directory containing the split folders ('train', 'valid', 'test').
    category : str
        The category from which to plot images (e.g., 'cars')
    split_type : str, optional
        The split type to pull images from ('train', 'valid', 'test'). Defaults to 'train'.
    num_to_plot : int, optional
        The number of images to plot. Defaults to 16.
    filetype : str, optional
        The file extension of the images to plot (e.g., 'png', 'jpg'). Defaults to 'png'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axes : array of matplotlib.axes
        The axes array containing the individual image subplots.
    """
    full_path = data_path / split_type / category
    
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} not found please check the path.")
    elif not any(full_path.glob(f"*.{filetype}")):
        raise FileNotFoundError(f"No images found in {full_path}. Please check your directory.")

    logging.info(f"Pulling samples from {full_path}")
    
    ncols = 4
    nrows = int(np.ceil(num_to_plot/ncols))
    
    # list files and select randomly if enough exist
    image_files = [f for f in os.listdir(full_path) if f.endswith('.' + filetype)]
    selected_files = random.sample(image_files, min(num_to_plot, len(image_files)))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(6, 1.5*nrows))

    for ax, img_file in zip(axes.flat, selected_files):
        # Load the image
        img = Image.open(full_path / img_file)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for cleaner display

    fig.suptitle(f"Category: {category}", y=0.97)
    fig.tight_layout()

    return fig, axes


def plot_batch(batch_images, batch_targets, category_map, max_to_plot=32, cmap='gray'):
    """
    Plots a batch of images, and their corresponding target categories, from a DataLoader.

    Parameters
    ----------
    batch_images : torch.Tensor
        A tensor containing a batch of images with shape `(N, C, H, W)`, where
        `N` is the batch size, `C` is the number of channels, `H` is the height,
        and `W` is the width of the images.
    batch_targets : torch.Tensor
        A tensor containing the target labels for the batch, with shape `(N,)`.
    category_map : dict
        A dictionary mapping category indices (as strings) to their human-readable
        labels, e.g., `{'0': 'car', '1': 'ant'}`.
    max_to_plot : int, optional
        The maximum number of images to plot from the batch. Defaults to 32.
    cmap : str, optional
        The colormap to use for displaying images. Defaults to 'gray'.

    Returns
    -------
    None
        Displays a grid of images with their corresponding labels.

    Notes
    -----
    - Images are converted to grayscale.
    - If batch size is smaller than `max_to_plot`, all images in batch will be plotted.
    """
    nbatch = len(batch_targets)
    num_to_plot = min(nbatch, max_to_plot)
    nrows = int(np.ceil(num_to_plot/4))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(7, 2*(nrows))) # size is width x height

    for index, ax in enumerate(axes.flat):
        if index >= num_to_plot:  # TODO: is this really needed?
            break
        image = batch_images[index]
        image = convert_for_plotting(image)
        category = str(batch_targets[index].item())
        ax.imshow(image, cmap=cmap)
        ax.set_title(category_map[category])
        ax.axis('off')

    fig.tight_layout()



def plot_transformed(original_image, transform, cmap=None, num_to_plot=4):
    """
    Plot the original image and pytorch transformations applied to it.

    original_image : 2d array-like image
        The original image to be transformed. Can be tensor or numpy/PIL or other array.
    transform : pytorch transform callable
        A transformation function (or series of transformations) to apply to the original image.
        The function should accept an image and return a transformed tensor.
    cmap : str, optional
        Colormap to use for displaying greyscale images. Set to None for color images.
    num_transforms : int, optional
        The number of transformed images to generate and display, in addition to original image. Defaults to 4.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axes : array of matplotlib.axes
        The axes array containing the individual image subplots.

    Notes
    -----
    - The first image displayed is the original, and subsequent images are transformed versions.
    """
    ncols = 5
    nrows = int(np.ceil((num_to_plot+1)/ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,2*nrows))

    for index, ax in enumerate(axes.flat):      
        if index >= num_to_plot+1:
            ax.axis('off')
            continue
        if index == 0:
            image = original_image
        else:
            image = convert_for_plotting(transform(original_image))  # convert for matplotlib
        ax.imshow(image)
        ax.axis('off')
        if index == 0:
            ax.set_title('Original')
        else:
            ax.set_title(f'Transform {index}')

    fig.tight_layout()

    return fig, axes
            

def convert_for_plotting(tensor):
    """
    Convert float torch tensor image to a uint8 tensor to format suitable for standard plotting libraries.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor image. Expected shape: (C, H, W). Typically a float, often not in [0, 1] range.

    Returns
    -------
    torch.Tensor
        A uint8 tensor image scaled to [0, 255] for plotting and dims (H,W,C)
    """
    # Validate input
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected input to be a torch.Tensor, but got {type(tensor)}.")
    if tensor.ndim != 3 or tensor.size(0) not in {1, 3}:
        raise ValueError(f"Expected tensor shape (1, H, W) or (3, H, W), but got {tuple(tensor.shape)}.")

    # Ensure the tensor is on CPU and detached from any computation graph
    tensor = tensor.detach().cpu()

    # Handle grayscale by expanding it to 3 channels for consistent plotting
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, -1, -1)  # Convert (1, H, W) -> (3, H, W)

    # Clamp float range to [0,1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    # Scale to [0, 255] and convert to uint8
    tensor = (tensor * 255).byte()

    # Reorder dimensions to (3, W, C) for standard plotting libraries
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)

    return tensor


def visualize_prediction(tensor, probabilities, category_map, top_n=5, logscale=False, axes=None, figsize=(5,3)):
    """
    Visualizes machine vision prediction by showing the image with the top predicted 
    label and a bar plot of the top N category probabilities.

    Parameters
    ----------
    tensor : torch.Tensor
        The input image tensor.
    probabilities : torch.Tensor
        The probabilities for each category (1D tensor).
    category_map : dict
        A dictionary mapping category index (as string) to category name.
        Example: {'0': 'dog', '1': 'cat'}
    top_n : int, optional
        Number of top categories to display, by default 5.
    logscale : bool, optional
        Whether to use a logarithmic scale for the bar plot, by default False.

    Returns
    -------
    axes: tuple of matplotlib.axes.Axes
        The axes objects for further customization or incorporation into larger plots.
    """

    # in case you got singleton batches
    probabilities = probabilities.squeeze(0)
    tensor = tensor.squeeze(0)  # Now shape is (C, H, W)
    
    # Ensure top_n doesn't exceed the number of available categories
    if top_n > len(category_map):
        logging.warning(f"top_n ({top_n}) is greater than the number of categories "
                        f"Setting top_n to {len(category_map)}.")
        top_n = len(category_map)
        

    # Get the top N probabilities and corresponding category indices
    top_probs, top_indices = torch.topk(probabilities, top_n)
    top_labels = [category_map[str(idx)] for idx in top_indices.cpu().numpy()]

    # Get tensor ready to plot
    tensor = tensor.squeeze(0)  # Now shape is (3, H, W)
    image = convert_for_plotting(tensor)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Display the image
    axes[0].imshow(image)
    axes[0].axis('off')
    # Add the top predicted label with confidence score
    predicted_label = top_labels[0]
    axes[0].set_title(f'{predicted_label} ({top_probs[0]:0.3f})')

    # Bar plot of probabilities
    xlabel = 'Log Probability' if logscale else 'Probability'
    axes[1].barh(top_labels, top_probs.cpu().numpy(), color='skyblue', log=logscale)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].invert_yaxis()  # Highest probability on top
    axes[1].set_title(f'Top {top_n} Predictions')

    plt.tight_layout()

    return axes
