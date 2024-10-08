# tests/test_training_utils.py
import torch

from deepglue import accuracy


def test_accuracy():
    # Example 1: Simple case with batch size 4 and 3 classes
    output = torch.tensor([
        [0.2, 0.5, 0.3],  # Image 1: Class 1 has highest score
        [0.1, 0.3, 0.6],  # Image 2: Class 2 has highest score
        [0.8, 0.1, 0.1],  # Image 3: Class 0 has highest score
        [0.4, 0.4, 0.2]   # Image 4: Tie between Class 0 and 1, picks 0 by default
    ])
    target = torch.tensor([1, 2, 0, 0])  # Ground truth labels

    # Test for top-1 accuracy
    top1_accuracy = accuracy(output, target, topk=(1,))
    assert top1_accuracy == [100.0], f"Expected [100.0] but got {top1_accuracy}"

    # Test for top-1 and top-2 accuracy
    topk_accuracy = accuracy(output, target, topk=(1, 2))
    assert topk_accuracy == [100.0, 100.0], f"Expected [100.0, 100.0] but got {topk_accuracy}"

    # Example 2: Another controlled case to check top-k behavior
    output = torch.tensor([
        [0.2, 0.7, 0.1],  # Image 1: Class 1 is correct, but Class 2 is in top-2
        [0.9, 0.05, 0.05],# Image 2: Class 0 correct, Class 1 also in top-2
        [0.1, 0.2, 0.7],  # Image 3: Class 2 correct
        [0.3, 0.6, 0.1]   # Image 4: Class 1 correct
    ])
    target = torch.tensor([1, 0, 2, 1])

    # Test for top-1 accuracy (should be 100%)
    top1_accuracy = accuracy(output, target, topk=(1,))
    assert top1_accuracy == [100.0], f"Expected [100.0] but got {top1_accuracy}"

    # Test for top-1 and top-2 accuracy (should be 100% and 100%)
    top2_accuracy = accuracy(output, target, topk=(1, 2))
    assert top2_accuracy == [100.0, 100.0], f"Expected [100.0, 100.0] but got {topk_accuracy}"

    # Example 3: A case with some incorrect predictions
    output = torch.tensor([
        [0.6, 0.2, 0.2],  # Image 1: Incorrect (should be 1)
        [0.1, 0.8, 0.1],  # Image 2: Correct (1)
        [0.1, 0.3, 0.6],  # Image 3: Correct (2)
        [0.7, 0.2, 0.1]   # Image 4: Incorrect (should be 1)
    ])
    target = torch.tensor([1, 1, 2, 1])

    # Test for top-1 and top-2 accuracy (should be 50% and 100%)
    top2_accuracy = accuracy(output, target, topk=(1, 2))
    assert top2_accuracy == [50.0, 100.0], f"Expected [50.0, 100.0] but got {topk_accuracy}"
