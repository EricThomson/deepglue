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

To inspect the different functions that are available, and their usage, see the [API page](api.md). 

## Demo notebook
Our [demo notebook](https://github.com/EricThomson/deepglue/tree/main/demos) walks you through deepglue's core features with a hands-on example. Highlights include:

- Setting up the project structure.
- Preparing datasets and data loaders.
- Defining and training a network.
- Visualizating feature clusters in the trained network.

More demos will be forthcoming as we continue adding features do deepglue (see the [roadmap](https://github.com/EricThomson/deepglue/issues/1)).
