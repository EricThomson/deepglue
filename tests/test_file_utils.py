# tests/test_file_utils.py
import logging
import pytest 
import shutil

from deepglue.file_utils import create_subdirs
from deepglue.file_utils import count_category_by_split
from deepglue.file_utils import count_by_category
from deepglue.file_utils import count_by_split
from deepglue.file_utils import sample_random_images

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



def test_count_category_by_split(setup_test_split_dirs):
    """
    Test that count_category_by_split correctly counts the number of images in each category.
    """
    data_path = setup_test_split_dirs
    counts = count_category_by_split(data_path)
    
    expected_counts = {
        'train': {'class0': 3, 'class1': 2},
        'valid': {'class0': 1, 'class1': 4},
        'test': {'class0': 0, 'class1': 3},
    }
    
    assert counts == expected_counts, "Counts do not match expected values."


def test_missing_split_directory(setup_test_split_dirs):
    """
    Test that count_category_by_split raises a FileNotFoundError when a split directory is missing.

    This test removes the 'valid' directory from the standard setup to simulate what happens
    when a part of the expected directory structure is missing.
    """
    data_path = setup_test_split_dirs

    # Simulate missing 'valid' directory by removing it along with its contents
    shutil.rmtree(data_path / 'valid')

    # Check that it raises expected error type
    with pytest.raises(FileNotFoundError):
        count_category_by_split(data_path)


def test_count_by_category(setup_test_split_dirs):
    """
    Test that count_by_category correctly calculates the total number of images
    for each category across all splits.
    """
    data_path = setup_test_split_dirs
    counts = count_by_category(data_path)

    expected_counts = {
        'class0': 4,  # 3 from train + 1 from valid + 0 from test
        'class1': 9   # 2 from train + 4 from valid + 3 from test
    }

    assert counts == expected_counts, "The category counts do not match the expected values."


def test_count_by_split(setup_test_split_dirs):
    """
    Test that get_samples_per_split correctly calculates the total number of samples
    in each split (train, valid, test) regardless of categories.
    """
    data_path = setup_test_split_dirs
    counts = count_by_split(data_path)

    expected_counts = {
        'train': 5,  # 3 images in class0 + 2 images in class1
        'valid': 5,  # 1 image in class0 + 4 images in class1
        'test': 3    # 0 images in class0 + 3 images in class1
    }

    assert counts == expected_counts, "The total sample counts per split do not match the expected values."


def test_sample_random_images_across_categories(setup_test_split_dirs):
    """Test sampling images in the train split."""
    data_path = setup_test_split_dirs
    category_map = {'class0': 'class0', 'class1': 'class1'}

    sampled_paths, sampled_categories = sample_random_images(data_path=data_path,
                                                             category_map=category_map,
                                                             num_images=4,
                                                             split_type='train')

    assert len(sampled_paths) == 4
    assert len(sampled_categories) == 4


def test_sample_random_images_from_category(setup_test_split_dirs):
    """Test sampling images from a specific category in the valid split."""
    data_path = setup_test_split_dirs
    category_map = {'class0': 'class0', 'class1': 'class1'}

    sampled_paths, sampled_categories = sample_random_images(data_path=data_path,
                                                             category_map=category_map,
                                                             num_images=2,
                                                             split_type='valid',

                                                             category='class1')
    # only 2 images were requested
    assert len(sampled_paths) == 2
    # makes sure category is the same for all sampled_categories
    assert all(category == 'class1' for category in sampled_categories)


def test_sample_random_images_exceeding_available(setup_test_split_dirs, caplog):
    """Test sampling more images than available with proper warning."""
    data_path = setup_test_split_dirs
    category_map = {'class0': 'class0', 'class1': 'class1'}

    with caplog.at_level(logging.WARNING):
        sampled_paths, _ = sample_random_images(data_path=data_path,
                                                category_map=category_map,
                                                num_images=10,  # Exceeding num available
                                                split_type='valid',
                                                category='class0')

    assert len(sampled_paths) == 1  # Only 1 image exists in valid/class0
    assert "Returning all available images" in caplog.text