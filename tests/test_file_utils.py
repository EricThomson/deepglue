# tests/test_file_utils.py
import pytest 
import shutil

from deepglue import create_subdirs
from deepglue import get_category_counts_by_split
from deepglue import get_samples_per_category
from deepglue import get_samples_per_split


@pytest.fixture
def setup_test_dirs_for_category_counts(tmp_path):
    """
    Sets up temporary test directories for train, valid, and test with sample images
    in different categories.

    Standard structure:
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
    tmp_path: Path
        The path to the temporary data directory.
    """
    # Create the train, valid, and test directories with class1 and class2
    split_types = ['train', 'valid', 'test']
    category_counts = {
        'class0': [3, 1, 0],  # Number of images in train, valid, test
        'class1': [2, 4, 3]
    }

    for split, counts in zip(split_types, zip(*category_counts.values())):
        split_dir = tmp_path / split
        split_dir.mkdir()
        for category, count in zip(category_counts.keys(), counts):
            category_dir = split_dir / category
            category_dir.mkdir()
            # Create the specified number of image files in each category
            for i in range(count):
                (category_dir / f'image_{i}.png').touch()

    return tmp_path


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



def test_get_category_counts_by_split(setup_test_dirs_for_category_counts):
    """
    Test that get_category_counts_by_split correctly counts the number of images in each category.
    """
    data_path = setup_test_dirs_for_category_counts
    counts = get_category_counts_by_split(data_path)
    
    expected_counts = {
        'train': {'class0': 3, 'class1': 2},
        'valid': {'class0': 1, 'class1': 4},
        'test': {'class0': 0, 'class1': 3},
    }
    
    assert counts == expected_counts, "Counts do not match expected values."


def test_missing_split_directory(setup_test_dirs_for_category_counts):
    """
    Test that get_category_counts_by_split raises a FileNotFoundError when a split directory is missing.

    This test removes the 'valid' directory from the standard setup to simulate what happens
    when a part of the expected directory structure is missing.
    """
    data_path = setup_test_dirs_for_category_counts

    # Simulate missing 'valid' directory by removing it along with its contents
    shutil.rmtree(data_path / 'valid')

    # Check that it raises expected error type
    with pytest.raises(FileNotFoundError):
        get_category_counts_by_split(data_path)


def test_get_samples_per_category(setup_test_dirs_for_category_counts):
    """
    Test that get_samples_per_category correctly calculates the total number of images
    for each category across all splits.
    """
    data_path = setup_test_dirs_for_category_counts
    counts = get_samples_per_category(data_path)

    expected_counts = {
        'class0': 4,  # 3 from train + 1 from valid + 0 from test
        'class1': 9   # 2 from train + 4 from valid + 3 from test
    }

    assert counts == expected_counts, "The category counts do not match the expected values."


def test_get_samples_per_split(setup_test_dirs_for_category_counts):
    """
    Test that get_samples_per_split correctly calculates the total number of samples
    in each split (train, valid, test) regardless of categories.
    """
    data_path = setup_test_dirs_for_category_counts
    counts = get_samples_per_split(data_path)

    expected_counts = {
        'train': 5,  # 3 images in class0 + 2 images in class1
        'valid': 5,  # 1 image in class0 + 4 images in class1
        'test': 3    # 0 images in class0 + 3 images in class1
    }

    assert counts == expected_counts, "The total sample counts per split do not match the expected values."
