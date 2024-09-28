# tests/test_file_utils.py
import pytest 
from pathlib import Path 

from deepglue import create_subdirs
from deepglue import get_category_counts_by_split


def test_create_subdirs(tmp_path):
    """"
    This uses pytest's built-in temporary fixture tmp_path 
    which is automatically cleaned up after the test runs
    All the testing for create_subdirs is done using this fixture.
    """
    # create temporary parent directory inside tmp_path
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    
    # Test suite 1: Test creating multiple subdirectories
    subdirs = ["subdir1", "subdir2"]
    created_paths = create_subdirs(parent_dir, subdirs)
    expected_paths = [parent_dir / "subdir1", parent_dir / "subdir2"]

    # first, check that the created paths match expectations
    assert created_paths == expected_paths
    # second, if so check to see that the paths actually exist
    assert all(path.exists() for path in created_paths)


    # Test suite 2: single string path
    created_paths = create_subdirs(parent_dir, subdirs[0])
    expected_path = [parent_dir / subdirs[0]]  # create_subdirs returns a list
    assert created_paths == expected_path
    assert created_paths[0].exists() # [0] because it returns a list

    # Test suite 3: test creating subdirectories when the parent directory does not exist
    new_parent_dir = tmp_path / "new_parent"
    subdirs = ["subdir3", "subdir4"]
    created_paths = create_subdirs(new_parent_dir, subdirs)
    expected_paths = [new_parent_dir / "subdir3", new_parent_dir / "subdir4"]

    # Check that the created paths match expectations
    assert created_paths == expected_paths
    # Check that the new parent directory and subdirectories exist
    assert new_parent_dir.exists()
    assert all(path.exists() for path in created_paths)


@pytest.fixture
def setup_test_image_category_dirs(tmp_path):
    """
    Sets up temporary test directories for train, valid, and test with standard directory structure for classes and images.

    Standard directory structure here is set up as:
        tmp_path/  # pytest fixture
            train/
                class0/   [3 images]
                class1/   [3 images]
                class2/   [3 images]
            valid/
                class0/   class1/   
            test/
                class0/   class1/   

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory unique to each test.

    Returns
    -------
    temp_path: Path
        The path to the temporary data directory.
    """
    # Create the train, valid, and test directories with class1 and class2
    split_types = ['train', 'valid', 'test']
    categories = ['class0', 'class1']

    for split in split_types:
        split_dir = tmp_path / split
        split_dir.mkdir()
        for category in categories:
            category_dir = split_dir / category
            category_dir.mkdir()
            # Create sample image files in each category
            for i in range(3):  # Create 3 images per category for simplicity
                (category_dir / f'image_{i}.png').touch()

    return tmp_path


def test_get_category_counts_by_split(setup_test_image_category_dirs):
    """
    Test that get_category_counts_by_split correctly counts the number of images in each category.
    """
    data_path = setup_test_image_category_dirs
    counts = get_category_counts_by_split(data_path)
    
    expected_counts = {
        'train': {'class0': 3, 'class1': 3},
        'valid': {'class0': 3, 'class1': 3},
        'test': {'class0': 3, 'class1': 3},
    }
    
    assert counts == expected_counts, "Counts do not match expected values."