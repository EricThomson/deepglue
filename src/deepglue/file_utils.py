"""
deepglue file_utils.py

This module contains functions for file management and handlling. 

Functions 
---------
create_subdirs(parent_dir, subdirs)
    creates subdirectories within given parent directory
"""
from pathlib import Path

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
    split_types = ['train', 'valid', 'test']
    num_category_by_split = {}
    
    for split_type in split_types:
        dataset_path = data_path / split_type
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Directory {dataset_path} does not exist. Please check your directory structure.")
        
        num_category_by_split[split_type] = {}
        
        for category in os.listdir(dataset_path):
            category_path = dataset_path / category
            if category_path.is_dir():
                num_images = len(list(category_path.glob('*'))) 
                num_category_by_split[split_type][category] = num_images

    return num_category_by_split
