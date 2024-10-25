"""
deep glue: A utility package for creating deep learning projects with pytorch.
"""

__version__ = "0.1.6"  

from .training_utils import train_one_epoch
from .training_utils import validate_one_epoch
from .training_utils import train_and_validate
from .training_utils import accuracy

from .plot_utils import plot_sample
from .plot_utils import plot_category_sample
from .plot_utils import plot_batch
from .plot_utils import plot_transformed
from .plot_utils import convert_for_plotting
from .plot_utils import visualize_prediction

from .file_utils import create_subdirs
from .file_utils import sample_random_images
from .file_utils import count_category_by_split
from .file_utils import count_by_category
from .file_utils import count_by_split


__all__ = ["train_one_epoch",
           "validate_one_epoch",
           "train_and_validate",
           "accuracy",
           
           "plot_sample",
           "plot_category_sample",
           "plot_batch",
           "plot_transformed",
           "convert_for_plotting",
           "visualize_prediction",
           
           "create_subdirs", 
           "sample_random_images",
           "count_category_by_split",
           "count_by_category",
           "count_by_split"]