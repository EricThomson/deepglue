"""
deepGlue: A utility package for deep learning projects.
"""

__version__ = "0.1.1"  

# Import functions to make them available at the package level
from .file_utils import create_subdirs

__all__ = ["create_subdirs"]