# tests/test_file_utils.py
from pathlib import Path
from src.deep_glue.file_utils import create_subdirs


def test_create_subdirs(tmp_path):
    # Use pytest's tmp_path fixture to create a temporary directory for testing
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    
    # Test creating multiple subdirectories
    subdirs = ["subdir1", "subdir2"]
    created_paths = create_subdirs(parent_dir, subdirs)
    assert all(path.exists() for path in created_paths)
    assert created_paths == [parent_dir / "subdir1", parent_dir / "subdir2"]

    # Test creating a single subdirectory
    single_subdir = "single_subdir"
    created_paths = create_subdirs(parent_dir, single_subdir)
    assert created_paths == [parent_dir / "single_subdir"]
    assert created_paths[0].exists()