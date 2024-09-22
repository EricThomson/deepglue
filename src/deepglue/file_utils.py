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