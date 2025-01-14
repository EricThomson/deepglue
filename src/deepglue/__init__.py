"""
deep glue: A utility package for creating deep learning projects with pytorch.
"""

__version__ = "0.1.8"  

from .training_utils import train_one_epoch
from .training_utils import validate_one_epoch
from .training_utils import train_and_validate
from .training_utils import predict_all 
from .training_utils import predict_batch
from .training_utils import accuracy
from .training_utils import extract_features
from .training_utils import prepare_ordered_data

from .plot_utils import plot_random_sample
from .plot_utils import plot_random_category_sample
from .plot_utils import plot_batch
from .plot_utils import plot_transformed
from .plot_utils import convert_for_plotting
from .plot_utils import plot_prediction_image
from .plot_utils import plot_prediction_probs
from .plot_utils import plot_prediction_grid
from .plot_utils import create_embeddable_image
from .plot_utils import plot_interactive_projection

from .file_utils import create_subdirs
from .file_utils import load_images_for_model
from .file_utils import sample_random_images
from .file_utils import count_category_by_split
from .file_utils import count_by_category
from .file_utils import count_by_split
from .file_utils import create_project 


__all__ = ["train_one_epoch",
           "validate_one_epoch",
           "train_and_validate",
           "predict_all",
           "predict_batch",
           "accuracy",
           "extract_features",
           "prepare_ordered_data",
           
           "plot_random_sample",
           "plot_random_category_sample",
           "plot_batch",
           "plot_transformed",
           "convert_for_plotting",
           "plot_prediction_image",
           "plot_prediction_probs",
           "plot_prediction_grid",
           "create_embeddable_image",
           "plot_interactive_projection",

           "create_subdirs", 
           "load_images_for_model",
           "sample_random_images",
           "count_category_by_split",
           "count_by_category",
           "count_by_split",
           "create_project"]