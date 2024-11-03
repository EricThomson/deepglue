# Deep Glue Usage Guide

Getting started with Deep Glue. 

## Installation

In the virtual environment in which you have torch installed:

    pip install deepglue

Or clone repo and do editable install with:

    pip install -e . 

## Usage

Import and use:

    import deepglue as dg

Then all functions will be available. For instance to train a model for one epoch:

    dg.train_one_epoch(model, data_loader, loss_function, optimizer)

To inspect the different functions that are available, and their usage, see the [API page](api.md). This is all in early stages -- in Spring 2025 we will provide some tutorials showing how to use Deep Glue to simplify machine vision tasks.  
