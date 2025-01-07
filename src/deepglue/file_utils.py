"""
deepglue file_utils.py

Module includes functions that are useful for wrangling directories and files. 
"""

import os
from pathlib import Path
from PIL import Image
import random
import torch

import logging
logging.getLogger(__name__)


def create_subdirs(parent_dir, subdirs):
    """
    Create subdirectories within a specified parent directory, unless they already exist.

    Parameters
    ---------
    parent_dir: str or Path
        The path to the parent directory where subdirectories will be created

    subdirs: list or str
        List of subdirectory names to create within the parent directory.
        If a single string is provided, will be converted to a list

    Returns
    -------
    new_paths: list of Path
        A list of path objects to the newly created subdirectories

    Example
    -------
    >>> create_subdirs(Path("path/to/parent"), ["subdir1", "subdir2"])
    [Path('/path/to/parent/subdir1'), Path('/path/to/parent/subdir2')]
    """
    logging.info(f"Creating subdirectories of {parent_dir}")

    parent_dir = Path(parent_dir)

    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)  # Create parent and intermediates if they don't exist

    # make sure subdirs is a list
    if isinstance(subdirs, str):
        subdirs = [subdirs]

    new_paths = []
    for subdir in subdirs:
        subdir_path = parent_dir / subdir
        if not subdir_path.exists():
            subdir_path.mkdir(parents=False, exist_ok=False) # prevent overwriting
        new_paths.append(subdir_path)

    return new_paths


def sample_random_images(data_path, category_map, num_images=1, split_type='train', category=None):
    """
    Randomly sample image paths from a dataset with a standard categorical directory structure.

    Parameters
    ----------
    data_path : str or Path
        Path to the root directory containing the split folders ('train', 'valid', 'test')
    category_map : dict
        Dictionary mapping category index (as string) to category name:  {'0': 'dog', '1': 'cat'}
    num_images : int, optional
        Number of image paths to sample, by default 1.
    split_type : str, optional
        The split folder to sample from ('train', 'valid', 'test'). Defaults to 'train'.
    category : str, optional
        If specified, only images from this category will be sampled. When default of None is
        chosen, will select randomly across all categories. 

    Returns
    -------
    sampled_paths: list
        len num_images list of paths to images
    sampled_categories: list
        len num_images list of corresponding categories 

    Raises
    ------
    FileNotFoundError
        If the specified split or category path does not exist.

    Notes
    -----
    - Assumes that each split folder contains only category subdirectories.
    - If `num_images` exceeds the total available images, all images will be returned, and a warning will be logged.
    """
    data_path = Path(data_path) 
    split_path =data_path / split_type
    logging.info(f"Selecting {num_images} random images from {data_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Split path {split_path} does not exist.")
    
    # Set up list of category directories (depends on whether single cat or all)
    if category is None:
        # filter out things that aren't directories
        category_dirs = [category_dir for category_dir in split_path.iterdir() if category_dir.is_dir()]
    else:
        # If a specific category is given, construct its path directly
        category_path = split_path / category
        if not category_path.is_dir():
            raise FileNotFoundError(f"Category directory name {category} does not exist in {split_path}.")
        category_dirs = [category_path]    # even though single category, expects list

    # Collect all image paths and their corresponding categories
    image_paths = []
    categories = []
    for category_dir in category_dirs:
        category_name = category_map[category_dir.name]
        for img_path in category_dir.glob('*'):
            image_paths.append(img_path)
            categories.append(category_name)

    total_images = len(image_paths)

    # Adjust num_images if it exceeds the available number of images
    if num_images > total_images:
        logging.warning(f"Requested {num_images} images, but only {total_images} are available. "
                        f"Returning all available images.")
        num_images = total_images

    # Randomly sample the requested number of images
    sampled_indices = random.sample(range(total_images), num_images)  # sample() works on iterable, so we give it range()
    sampled_paths = [image_paths[i] for i in sampled_indices]
    sampled_categories = [categories[i] for i in sampled_indices]

    return sampled_paths, sampled_categories


def load_images_for_model(image_paths, transform):
    """
    Given a list of image paths, returns a tensor suitable for model input.

    Parameters
    ----------
    image_paths : list of str or Paths
        List of image file paths.
    transform : torchvision transform (callable)
        The transformations to apply to each image.

    Returns
    -------
    torch.Tensor
        A batch of images as a tensor of shape (len(image_paths), 3, H, W).
    """
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    return torch.stack(images)


def count_by_category(data_path):
    """
    Calculates the total number of images for each category across all splits.

    Traverses the train, valid, and test folders and aggregates image counts
    for each category. This can be useful for identifying category imbalances.

    Parameters
    ----------
    data_path : Path
        The path to the parent directory containing the 'train', 'valid', and 'test' folders. They each
        contain the same category-specific subdirectories.
        
    Returns
    -------
    num_per_category : dict
        A dictionary where keys are category names and values are the total number of images in each category.

    Raises
    ------
    FileNotFoundError
        If any of the specified split directories ('train', 'valid', 'test') do not exist at the given path.
    """
    logging.info(f"Getting samples per category in {data_path}")

    num_per_category = {}
    split_types = ['train', 'valid', 'test']

    for split_type in split_types:
        split_path = data_path / split_type

        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} does not exist. Please check your directory structure.")

        # Traverse each category directory in the split
        for category_path in split_path.iterdir():
            if category_path.is_dir():
                # Use glob to count files in each category directory
                category_name = category_path.name
                num_images = len(list(category_path.glob('*')))
                # initialize
                if category_name not in num_per_category:
                    num_per_category[category_name] = 0
                num_per_category[category_name] += num_images

    return num_per_category


def count_by_split(data_path):
    """
    Calculates the total number of images in train, test, and validation splits, regardless of categories.

    This function directly traverses the 'train', 'valid', and 'test' folders and counts all image files,
    providing the total number of samples in each split without considering category distinctions.

    Parameters
    ----------
    data_path : Path
        The path to the directory containing the 'train', 'valid', and 'test' folders.

    Returns
    -------
    num_per_split : dict
        A dictionary with keys 'train', 'valid', and 'test', each containing the total number of samples 
        in each split, regardless of category.

    Raises
    ------
    FileNotFoundError
        If any of the specified split directories ('train', 'valid', 'test') do not exist at the given path.
    """
    logging.info(f"Getting samples per split in {data_path}")

    num_per_split = {}
    split_types = ['train', 'valid', 'test']

    for split_type in split_types:
        split_path = data_path / split_type

        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} does not exist. Please check your directory structure.")

        split_total = 0
        # Directly iterate over category directories within the split folder
        for category_dir in split_path.iterdir():
            if category_dir.is_dir():
                # Count files in the category directory
                split_total += len(list(category_dir.glob('*')))

        num_per_split[split_type] = split_total

    return num_per_split


def count_category_by_split(data_path):
    """
    Counts the number of images in each category within train, validation, and test splits.

    Assumes a directory structure where images are stored in category-specific 
    subdirectories under 'train', 'valid', and 'test' folders.

    Parameters
    ----------
    data_path : Path
        The path to the directory containing the 'train', 'valid', and 'test' folders.

    Returns
    -------
    num_category_per_split: dict
        A nested dictionary with keys 'train', 'valid', and 'test', each containing a sub-dictionary 
        where the keys are category names and the values are the counts of images in each category.

    Raises
    ------
    FileNotFoundError
        If any of the 'train', 'valid', or 'test' directories do not exist at the specified path.
    """
    logging.info(f"Getting category counts by split in {data_path}")

    split_types = ['train', 'valid', 'test']
    num_category_per_split = {}
    
    for split_type in split_types:
        dataset_path = data_path / split_type
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"{dataset_path} does not exist. Please check your directory structure.")
        
        num_category_per_split[split_type] = {}
        
        for category in os.listdir(dataset_path):
            category_path = dataset_path / category
            if category_path.is_dir():
                num_images = len(list(category_path.glob('*'))) 
                num_category_per_split[split_type][category] = num_images

    return num_category_per_split


def create_project(projects_dir, project_name):
    """
    Creates a minimal project directory structure within the project parent directory:
    
        projects_dir/
            project_name/
                data/
                models/

    Parameters
    ----------
    projects_dir : str or Path
        Path to the project parent directory. 
    project_name : str
        name of the project (must be a valid directory name: avoid spaces and other weird things)

    Returns
    -------
    project_dir : Path
        Path to the project directory that was created in projects_dir
    data_dir : Path
        Path to the data directory in project_dir
    models_dir : Path
        Path to the models directory in the project_dir

    TODO
    ----
    consider using pathvalidate to throw error if project_name is invalid
    """
    projects_dir = Path(projects_dir)
    project_dir = projects_dir / project_name

    try:
        project_dir.mkdir(parents=True, exist_ok=False)
        logging.info(f"Created project directory: {project_dir}")
    except FileExistsError:
        logging.info(f"Project directory '{project_dir}' already exists. Skipping.")

    subdirs = ["data", "models"]
    project_subdirs = create_subdirs(project_dir, subdirs)
    data_dir, models_dir = project_subdirs[0], project_subdirs[1]

    return project_dir, data_dir, models_dir