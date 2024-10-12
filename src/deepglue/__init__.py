"""
deepGlue: A utility package for deep learning projects.
"""

__version__ = "0.1.2"  

from .training_utils import train_one_epoch
from .training_utils import validate_one_epoch
from .training_utils import train_and_validate
from .training_utils import accuracy

from .plot_utils import plot_category_samples
from .plot_utils import plot_batch
from .plot_utils import plot_transformed

from .file_utils import create_subdirs
from .file_utils import get_category_counts_by_split
from .file_utils import get_samples_per_category
from .file_utils import get_samples_per_split


__all__ = ["train_one_epoch",
           "validate_one_epoch",
           "train_and_validate",
           "accuracy",
           
           "plot_category_samples",
           "plot_batch",
           "plot_transformed",
           
           "create_subdirs", 
           "get_category_counts_by_split",
           "get_samples_per_category",
           "get_samples_per_split"]