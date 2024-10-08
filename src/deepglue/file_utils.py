"""
deepglue file_utils.py

Module includes functions that are useful for wrangling directories and files. 
"""

import os
from pathlib import Path

import logging
logging.getLogger(__name__)

def create_subdirs(parent_dir, subdirs):
    """
    Create subdirectories within a specified parent directory

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


def get_category_counts_by_split(data_path):
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
    num_category_by_split: dict
        A nested dictionary with keys 'train', 'valid', and 'test', each containing a sub-dictionary 
        where the keys are category names and the values are the counts of images in each category.

    Raises
    ------
    FileNotFoundError
        If any of the 'train', 'valid', or 'test' directories do not exist at the specified path.
    """
    logging.info(f"Getting category counts by split in {data_path}")

    split_types = ['train', 'valid', 'test']
    num_category_by_split = {}
    
    for split_type in split_types:
        dataset_path = data_path / split_type
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"{dataset_path} does not exist. Please check your directory structure.")
        
        num_category_by_split[split_type] = {}
        
        for category in os.listdir(dataset_path):
            category_path = dataset_path / category
            if category_path.is_dir():
                num_images = len(list(category_path.glob('*'))) 
                num_category_by_split[split_type][category] = num_images

    return num_category_by_split


def get_samples_per_category(data_path):
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
    samples_per_category : dict
        A dictionary where keys are category names and values are the total number of images in each category.

    Raises
    ------
    FileNotFoundError
        If any of the specified split directories ('train', 'valid', 'test') do not exist at the given path.
    """
    logging.info(f"Getting samples per category in {data_path}")

    samples_per_category = {}
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
                if category_name not in samples_per_category:
                    samples_per_category[category_name] = 0
                samples_per_category[category_name] += num_images

    return samples_per_category


def get_samples_per_split(data_path):
    """
    Calculates the total number of samples in train, test, and validation splits, regardless of categories.

    This function directly traverses the 'train', 'valid', and 'test' folders and counts all image files,
    providing the total number of samples in each split without considering category distinctions.

    Parameters
    ----------
    data_path : Path
        The path to the directory containing the 'train', 'valid', and 'test' folders.

    Returns
    -------
    samples_per_split : dict
        A dictionary with keys 'train', 'valid', and 'test', each containing the total number of samples 
        in each split, regardless of category.

    Raises
    ------
    FileNotFoundError
        If any of the specified split directories ('train', 'valid', 'test') do not exist at the given path.
    """
    logging.info(f"Getting samples per split in {data_path}")

    samples_per_split = {}
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

        samples_per_split[split_type] = split_total

    return samples_per_split