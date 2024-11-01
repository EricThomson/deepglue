"""
deepglue plot_utils.py

Module includes functions that are useful for plotting/visualization during different
deep learning tasks
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from deepglue.file_utils import sample_random_images


# Initialize things
logging.getLogger(__name__)

# Set global style for all plots when the module is imported
plt.rcParams.update({'figure.titlesize': 14, # suptitle 
                     'axes.titlesize': 12, # individual plot titles
                    'axes.labelsize': 12,  # x and y labels
                    'legend.fontsize': 10.5, # legend labels 
                    'xtick.labelsize': 10, # x-tick labels 
                    'ytick.labelsize': 10})  


def plot_random_sample(data_path, category_map, split_type='train', num_to_plot=16):
    """
    Plots random image samples from a specified data split.

    Assumes a directory structure where images are stored in category-specific
    subdirectories inside the split folders ('train', 'valid', 'test').

        data_path/
            train/
                cat/
                dog/
            valid/
                cat/
                dog/   
            test/
                cat/
                dog/

    Parameters
    ----------
    data_path : str or Path
        The path to the root directory containing the split folders ('train', 'valid', 'test').
    category_map : dict
        A dictionary mapping category indices (as strings) to their human-readable
        labels, e.g., `{'0': 'cat', '1': 'dog'}`.
    split_type : str, optional
        The split folder to pull images from ('train', 'valid', 'test'). Defaults to 'train'.
    num_to_plot : int, optional
        Number of images to plot. Defaults to 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots
    axes : array of matplotlib.axes
        An array of matplotlib Axes objects, one for each image subplot.
    """
    sample_paths, sample_categories = sample_random_images(data_path, 
                                                           category_map, 
                                                           split_type=split_type, 
                                                           num_images=num_to_plot)
    ncols = 4
    nrows = int(np.ceil(num_to_plot/ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(6, 1.5*nrows))
    
    for ax, sample_path, sample_category in zip(axes.flat, sample_paths, sample_categories):
        # Load the image
        img = Image.open(sample_path)
        ax.imshow(img)
        ax.set_title(sample_category)
        ax.axis('off') 
    
    fig.suptitle(f"Random images from {split_type} split", y=0.96)
    fig.tight_layout()
    
    return fig, axes


def plot_random_category_sample(data_path, category, split_type='train', num_to_plot=16):
    """
    Plots a random selection of images from a specific category within a data split.

    Assumes a directory structure where images are stored in category-specific 
    subdirectories under split folders (e.g., 'train', 'valid', 'test'):

    Parameters
    ----------
    data_path : str or Path
        The path to the root directory containing the split folders ('train', 'valid', 'test').
    category : str
        The name of the category from which to plot images (e.g., 'cat')
    split_type : str, optional
        The split folder to pull images from ('train', 'valid', 'test'). Defaults to 'train'.
    num_to_plot : int, optional
        The number of images to plot. Defaults to 16. If it exceeds the available number of images,
        a warning will be issued and all available images will be plotted. 

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axes : array of matplotlib.axes
        An array of matplotlib Axes objects, one for each image subplot.
    """
    data_path = Path(data_path) # in case it's a string

    # make a dummy category map for sample_random_images() to work with
    category_map = {category: category}
    
    # Use dg.sample_random_images() to select the images from the specified category
    sampled_paths, _ = sample_random_images(data_path=data_path,
                                            category_map=category_map,
                                            num_images=num_to_plot,
                                            split_type=split_type,
                                            category=category)

    if not sampled_paths:
        raise FileNotFoundError(f"No images found in '{split_type}/{category}'.")

    num_to_plot = len(sampled_paths) # in case too manhy requested

    ncols = 4
    nrows = int(np.ceil(num_to_plot/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(6, 1.5*nrows))

    for ax, img_file in zip(axes.flat, sampled_paths):
        # Load the image
        img = Image.open(img_file)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for cleaner display

    # Hide any unused axes
    for ax in axes.flat[num_to_plot:]:
        ax.axis('off')

    fig.suptitle(f"Category: {category} ({split_type} split)", y=0.97)
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


def plot_prediction_image(tensor, probabilities, category_map, 
                               true_label=None, ax=None, figsize=(2.5, 2.5)):
    """
    Plot classifier prediction: displays image with true label on top and estimate on bottom with probability.

    Parameters
    ----------
    tensor : torch.Tensor
        The input image tensor (CxHxW) or 1xCxHxW
    probabilities : torch.Tensor
        Prediction probabilities for each category (1D tensor).
    category_map : dict
        A mapping of category indices (as strings) to their respective labels.
        Example: {'0': 'cat', '1': 'dog'}.
    true_label : str, optional
        The actual category label of the image, if known (e.g., 'dog'). Default is None.
    axes : matplotlib.axes.Axes, optional
        Axes object for plot. If None, new axes are created. Default is None.
    figsize : tuple, optional
        Size of the figure in inches. Default is (2.5, 2.5).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further customization or saving.
    axes : matplotlib.axes.Axes
        The image axis object
    """
    # Handle any stray tensor dimensions
    probabilities = probabilities.squeeze(0)
    tensor = tensor.squeeze(0)

    # Get top predictions
    top_prob, top_ind = torch.topk(probabilities, 1)
    predicted_label = category_map[str(top_ind.item())]
    image = convert_for_plotting(tensor)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = plt.gcf()

    # Plot the image
    ax.imshow(image)
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(f"Est: {predicted_label} ({top_prob.item():.2f})")  
    if true_label:
        ax.set_title(f"{true_label}")   
    
    return fig, ax


def plot_prediction_probs(probabilities, category_map, true_label=None, top_n=5, logscale=True, 
                               ax=None, figsize=(3, 2.5), bar_color='skyblue'):
    """
    Plot classifier prediction probabilities: bar plot of top N category probabilities.

    Parameters
    ----------
    probabilities : torch.Tensor
        Prediction probabilities for each category (1D tensor).
    category_map : dict
        A mapping of category indices (as strings) to their respective labels.
        Example: {'0': 'cat', '1': 'dog'}.
    true_label : str, optional
        The actual category label, if known (e.g., 'dog'). Default is None.
    top_n : int, optional
        The top n class probabilities to display from the classifier, default is 5.
    logscale : bool, optional
        If True, the bar plot uses a logarithmic scale. Default is True.
    axes : matplotlib.axes.Axes, optional
        Axes object for plot. If None, new axes are created. Default is None.
    figsize : tuple, optional
        Size of the figure in inches. Default is (2.5, 2.5).
    bar_color : str, optional
        Color for the bars in the bar plot. Default is 'skyblue'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further customization or saving.
    axes : matplotlib.axes.Axes
        The bar plot axis object

    """
    # Handle any stray tensor dimensions
    probabilities = probabilities.squeeze(0)

    # Ensure top_n doesn't exceed the number of available categories
    if top_n > len(category_map):
        logging.warning(f"top_n ({top_n}) is greater than the number of categories "
                        f"Setting top_n to {len(category_map)}.")
        top_n = len(category_map)
        
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, top_n)
    top_labels = [category_map[str(idx)] for idx in top_indices.cpu().numpy()]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = plt.gcf()

    # Plot the bar chart of top n probabilities
    ax.barh(top_labels, top_probs.cpu().numpy(), color=bar_color, log=logscale)
    ax.set_xlabel("Log Probability" if logscale else "Probability")
    ax.invert_yaxis() # high on top
    ax.set_title(f"Top {top_n} Predictions")

    # Set y-tick labels with bold formatting for the true label
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        if label.get_text() == true_label:
            label.set_fontweight('bold')  # Set correct label to bold

    return fig, ax


def plot_prediction_grid(images, probability_matrix, true_categories, category_map, 
                          top_n=5, figsize_per_plot=(2, 3), logscale=True):
    """
    Plots a grid of classifier prediction visualizations.
    
    Each visualization in the grid contains the image on the left , plotted 
    using dg.plot_prediction_image() and bar plot of top_n category 
    probabilities on the right, plotting using dg.visualize_prediction_probs()

    Parameters
    ----------
    images : torch.Tensor
        Shape num_predictions x 3 x H x W of images to be classified
    probability_matrix : torch.Tensor
        Torch tensor w/shape num predictions x num categories
        Each row corresponds to image and contains classifier probabilities for each category.
    true_categories : list of str
        Length num_predictions list of correct labels for each prediction (e.g., ['cat', 'dog'...]
    category_map : dict
        A mapping of category indices (as strings) to their respective labels.
        Example: {'0': 'cat', '1': 'dog'}.
    top_n : int, optional
        The top n class probabilities to show in bar plot, default is 5.
    figsize_per_plot : tuple, optional
        Size of each (image + bar plot) pair in inches. Default is (3, 3).
    logscale : bool, optional
        If True, the bar plot uses a logarithmic scale. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the full grid of prediction plots.
    axes : np.ndarray of matplotlib.axes.Axes
        Array of axes objects arranged in a grid

    Note
    ----
    Inspired by similar visualization created by the Nuevo Foundation:
    https://workshops.nuevofoundation.org/python-tensorflow/plotting_model/
    """
    num_predictions = len(true_categories)

    if images.shape[0] != num_predictions or probability_matrix.shape[0] != num_predictions:
        raise ValueError("The length of `true_categories` must match the first dimension of `images` and `probability_matrix`.")
    
    predictions_per_row = 2
    subplots_per_prediction = 2  # Image + bar plot 
    ncols = predictions_per_row * subplots_per_prediction
    nrows = int(np.ceil(num_predictions / predictions_per_row))
    
    # Create figure with specified grid layout
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, 
                                                    figsize_per_plot[1] * nrows), 
                             layout="constrained")

    for prediction_ind in range(num_predictions):
        # First work out indexing and assign to image and bar plot
        row, position_in_row = divmod(prediction_ind, predictions_per_row)
        col_start = position_in_row * 2  # Start column for this prediction group
        im_ax = axes[row, col_start]
        prob_ax = axes[row, col_start + 1]

        # Get true label for the current prediction
        true_label = true_categories[prediction_ind]

        # Plot the image with the actual and predicted label
        _, im_ax = plot_prediction_image(images[prediction_ind], 
                                                 probability_matrix[prediction_ind], 
                                                 category_map, 
                                                 true_label=true_label, 
                                                 ax=im_ax)

        # Plot the bar plot and remove title and xlabel
        _, prob_ax = plot_prediction_probs(probability_matrix[prediction_ind], 
                                                   category_map, 
                                                   true_label=true_label, 
                                                   top_n=top_n, 
                                                   logscale=logscale, 
                                                   ax=prob_ax)
        prob_ax.set_title("")
        prob_ax.set_xlabel("")

    # Remove unused axes 
    for unused_ax in axes.flatten()[num_predictions * subplots_per_prediction:]:
        unused_ax.axis("off")
        
    return fig, axes