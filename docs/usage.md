# Deep Glue Usage Guide

Getting started with Deep Glue. 

## Installation

In the virtual environment in which you have torch installed:

    pip install deepglue

Or clone repo and do dev install with:

    pip install -e . 

## Usage

Import and use:

    import deepglue as dg

Then all functions will be available. For instance to train a model for one epoch:

    dg.train_one_epoch(model, data_loader, loss_function, optimizer)

This is all in early stages -- eventually there will be simple tutorials to show how to use Deep Glue for classification tasks, etc.  

## More details

To inspect the different functions and their usage in more detail, see the [API page](api.md).
