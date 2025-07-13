# tests/test_file_utils.py
import logging
import pytest 
import shutil

from deepglue.file_utils import create_subdirs
from deepglue.file_utils import count_category_by_split
from deepglue.file_utils import count_by_category
from deepglue.file_utils import count_by_split
from deepglue.file_utils import sample_random_images
from deepglue.file_utils import split_dataset
from deepglue.file_utils import create_project

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



def test_count_category_by_split(setup_test_dataset):
    """
    Test that count_category_by_split correctly counts the number of images in each category.
    """
    data_path = setup_test_dataset
    counts = count_category_by_split(data_path)
    
    expected_counts = {
        'train': {'class0': 3, 'class1': 2},
        'valid': {'class0': 1, 'class1': 4},
        'test': {'class0': 0, 'class1': 3},
    }
    
    assert counts == expected_counts, "Counts do not match expected values."


def test_missing_split_directory(setup_test_dataset):
    """
    Test that count_category_by_split raises a FileNotFoundError when a split directory is missing.

    This test removes the 'valid' directory from the standard setup to simulate what happens
    when a part of the expected directory structure is missing.
    """
    data_path = setup_test_dataset

    # Simulate missing 'valid' directory by removing it along with its contents
    shutil.rmtree(data_path / 'valid')

    # Check that it raises expected error type
    with pytest.raises(FileNotFoundError):
        count_category_by_split(data_path)


def test_count_by_category(setup_test_dataset):
    """
    Test that count_by_category correctly calculates the total number of images
    for each category across all splits.
    """
    data_path = setup_test_dataset
    counts = count_by_category(data_path)

    expected_counts = {
        'class0': 4,  # 3 from train + 1 from valid + 0 from test
        'class1': 9   # 2 from train + 4 from valid + 3 from test
    }

    assert counts == expected_counts, "The category counts do not match the expected values."


def test_count_by_split(setup_test_dataset):
    """
    Test that get_samples_per_split correctly calculates the total number of samples
    in each split (train, valid, test) regardless of categories.
    """
    data_path = setup_test_dataset
    counts = count_by_split(data_path)

    expected_counts = {
        'train': 5,  # 3 images in class0 + 2 images in class1
        'valid': 5,  # 1 image in class0 + 4 images in class1
        'test': 3    # 0 images in class0 + 3 images in class1
    }

    assert counts == expected_counts, "The total sample counts per split do not match the expected values."


def test_sample_random_images_across_categories(setup_test_dataset):
    """
    Test sampling images in the train split.
    """
    data_path = setup_test_dataset
    category_map = {'class0': 'class0', 'class1': 'class1'}

    sampled_paths, sampled_categories = sample_random_images(data_path=data_path,
                                                             category_map=category_map,
                                                             num_images=4,
                                                             split_type='train')

    assert len(sampled_paths) == 4
    assert len(sampled_categories) == 4


def test_sample_random_images_from_category(setup_test_dataset):
    """
    Test sampling images from a specific category in the valid split.
    """
    data_path = setup_test_dataset
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


def test_sample_random_images_exceeding_available(setup_test_dataset, caplog):
    """Test sampling more images than available with proper warning."""
    data_path = setup_test_dataset
    category_map = {'class0': 'class0', 'class1': 'class1'}

    with caplog.at_level(logging.WARNING):
        sampled_paths, _ = sample_random_images(data_path=data_path,
                                                category_map=category_map,
                                                num_images=10,  # Exceeding num available
                                                split_type='valid',
                                                category='class0')

    assert len(sampled_paths) == 1  # Only 1 image exists in valid/class0
    assert "Returning all available images" in caplog.text


def test_split_dataset(tmp_path):
    """
    Happy path test for split_dataset function
    Raw data will be in raw_data. Splits will be put into split_data with 60/20/20 split
    """
    # Create initial raw data directories
    raw_data = tmp_path / "raw_data"
    cat_dir = raw_data / "cat"
    dog_dir = raw_data / "dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir()

    # Create initial raw data files and make a record of their filenames
    all_files_by_category = {"cat": [],
                             "dog": []}
    
    for i in range(10):
        filename = f"cat_{i:02d}.jpg"
        (cat_dir / filename).touch()
        all_files_by_category["cat"].append(filename)

    for i in range(20):
        filename = f"dog_{i:02d}.jpg"
        (dog_dir / filename).touch()
        all_files_by_category["dog"].append(filename)


    target_dir = tmp_path / "split_data"


    # set up expected results (counts and file structure)
    expected_counts = {"train": {"cat": 6, "dog": 12},
                       "valid": {"cat": 2, "dog": 4},
                       "test": {"cat": 2, "dog": 4}}
    
    # create expected set of files by split 
    expected_files_by_split = {"train": {},
                               "valid": {},
                               "test": {}}
    
    categories = ["cat", "dog"]
    splits = ["train", "valid", "test"]
    for category in categories:
        all_files = all_files_by_category[category]
        current = 0

        for split in splits:
            count = expected_counts[split][category]
            files_for_split = all_files[current : current + count]
            expected_files_by_split[split][category] = set(files_for_split)
            current += count
    

    # Run the split
    counts = split_dataset(source_dir=raw_data,
                           target_dir=target_dir,
                           splits=(0.6, 0.2, 0.2),
                           shuffle=False)

    # Check returned counts are as expected
    assert counts == expected_counts, f"Counts mismatch!\nExpected: {expected_counts}\nActual: {counts}"

    # Check that directories and files were created correctly
    for split in splits:
        for category in categories:
            split_cat_dir = target_dir / split / category

            assert split_cat_dir.exists(), f"Missing folder: {split_cat_dir}"

            files = list(split_cat_dir.iterdir())
            actual_files = {f.name for f in files}
            expected_files = expected_files_by_split[split][category]

            assert actual_files == expected_files, (
                f"Expected and actual files in {split_cat_dir} do not match.\n"
                f"Expected: {expected_files}\n"
                f"Actual: {actual_files}"
            )

    # Check original files still exist (didn't get moved): sets ignore order
    expected_cat_files = {f"cat_{i:02d}.jpg" for i in range(10)}
    actual_cat_files = {f.name for f in (raw_data / "cat").iterdir()}
    assert actual_cat_files == expected_cat_files, "Original cat files missing or changed."

    expected_dog_files = {f"dog_{i:02d}.jpg" for i in range(20)}
    actual_dog_files = {f.name for f in (raw_data / "dog").iterdir()}
    assert actual_dog_files == expected_dog_files, "Original dog files missing or changed."


def test_split_dataset_missing_source(tmp_path):
    """
    Test that split_dataset raises FileNotFoundError when source_dir doesn't exist.
    """
    fake_source = tmp_path / "does_not_exist"
    target = tmp_path / "split_data"

    with pytest.raises(FileNotFoundError):
        split_dataset(source_dir=fake_source,
                      target_dir=target)


def test_split_dataset_invalid_splits(tmp_path):
    """
    Test that split_dataset raises ValueError when splits don't sum to 1.0.
    """
    # create directory so we won't get FIleNotFoundError (see previous test) 
    raw_data = tmp_path / "raw_data" 
    raw_data.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        split_dataset(source_dir=raw_data,
                      target_dir=tmp_path / "split_data",
                      splits=(0.5, 0.3, 0.3),  # sums to 1.1
                      shuffle=False)


def test_create_project(tmp_path):
    """
    Test the create_project function to ensure it creates the correct directory structure 
    and that if it is run multiple times the same structure is created/maintained.
    """
    # Define test parameters
    projects_dir = tmp_path / "projects"
    project_name = "test_project"

    # Call the function
    project_dir, data_dir, models_dir = create_project(projects_dir, project_name)

    # Define expected structure
    expected_project_dir = projects_dir / project_name
    expected_data_dir = project_dir / "data"
    expected_models_dir = project_dir / "models"

    # Check that the project directory and children exist
    assert project_dir.exists(), f"Project directory {project_dir} was not created."
    assert data_dir.exists(), f"Data directory {data_dir} was not created."
    assert models_dir.exists(), f"Models directory {models_dir} was not created."

    # Check that returned paths match the expected subdirectories
    assert project_dir == expected_project_dir, "Returned project_dir does not match the expected path."
    assert data_dir == expected_data_dir, "Returned data_dir does not match the expected path."
    assert models_dir == expected_models_dir, "Returned models_dir does not match the expected path."

    # Ensure re-run doesn't fail or overwrite existing structure
    project_dir_again, data_dir_again, models_dir_again = create_project(projects_dir, project_name)
    assert project_dir_again == expected_project_dir, "Re-running create_project altered project_dir structure."
    assert data_dir_again == expected_data_dir, "Re-running create_project altered data_dir structure."
    assert models_dir_again == expected_models_dir, "Re-running create_project altered models_dir structure."



def test_create_project_preserves_data(tmp_path):
    """
    Make sure create_project does not overwrite existing data in a project directory.
    Basically, a more stringent overwrite test than in test_create_project. 

    TODO: 
        test that it not only keeps the files, but preserves data (e.g., doesn't create empty file). 
    """
    # Create a project directory with dummy data
    projects_dir = tmp_path / "projects"
    project_name = "test_project"
    project_dir, data_dir, models_dir = create_project(projects_dir, project_name)

    # Add dummy data to the 'data' directory
    data_dir = project_dir / "data"
    dummy_file = data_dir / "dummy.txt"
    dummy_file.write_text("This is a test file.")

    # Ensure the dummy file exists with the correct content
    assert dummy_file.exists(), "Dummy file was not created."
    assert dummy_file.read_text() == "This is a test file."

    # Re-run the function
    project_dir_again, data_dir_again, models_dir_again = create_project(projects_dir, project_name)

    # Verify the dummy file still exists and is unmodified
    assert dummy_file.exists(), "Dummy file was overwritten or deleted."
    assert dummy_file.read_text() == "This is a test file.", "Dummy file content was modified."

    # Verify the structure is maintained
    assert data_dir.exists(), "Data directory was not preserved."
    assert models_dir.exists(), "Models directory was not preserved."
    assert project_dir_again == project_dir, "Project directory path was altered on re-run."
    assert data_dir_again == data_dir, "Data directory path was altered on re-run."
    assert models_dir_again == models_dir, "Models directory path was altered on re-run."
