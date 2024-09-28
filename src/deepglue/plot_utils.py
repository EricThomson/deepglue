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
        The array of axes objects in the plot.
    """
    full_path = data_path / split_type / category
    
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} not found please check the path.")
    elif not any(full_path.glob(f"*.{filetype}")):
        raise FileNotFoundError(f"No images found in {full_path}. Please check your directory.")

    print(f"Pulling random images from {full_path}")
    
    ncols = 4
    nrows = int(np.ceil(num_to_plot/ncols))
    
    # list files and select randomly if enough exist
    image_files = [f for f in os.listdir(full_path) if f.endswith('.' + filetype)]
    selected_files = random.sample(image_files, min(num_to_plot, len(image_files)))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(6, 1.5*nrows))

    for ax, img_file in zip(axes.flat, selected_files):
        # Load the image
        img = Image.open(full_path / img_file)
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Hide axes for cleaner display

    fig.suptitle(f"Category: {category}", fontsize=16, y=0.97)
    fig.tight_layout()

    return fig, axes