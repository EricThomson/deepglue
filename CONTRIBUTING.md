
# Contributing to deepglue

<img src="https://raw.githubusercontent.com/EricThomson/deepglue/main/docs/images/deep_glue_logo.png" alt="deepglue logo" align="right" width="125">

There are many different ways you can contribute to deepglue, from sharing ideas in discussions or issues, to helping improve documentation and the code base. Any help is welcome and appreciated!

## Ways to contribute

The easiest way to contribute is just to ask questions by opening an issue or discussion. If you find problems or bugs, or just have a question, feel free to ask. Also, the documentation is always a work in progress. Any help making it more clear, or keeping it up-to-date, is appreciated. This includes the demo notebooks. If you want to contribute code, you can open a pull request.

## How to make a pull request
Below is a brief summary of how to make a pull request (PR) to deepglue: if you have never done this before, please reach out and I'll help you through the process. 

1. **Fork the Repository**: Click the "Fork" button at the top right of the deepglue repo.
2. **Clone Your Fork**: Clone your fork to your local machine.
   ```
   git clone https://github.com/your-username/deepglue.git
   ```
3. **Set up your dev environment**: Install dependencies and set up environment using pip (or uv).
   ```
   uv pip install -e .[dev,docs]
   ```
4. **Create a Branch**: Create a new branch for your work.
   ```
   git checkout -b my-new-branch
   ```
5. **Make Your Changes**: Make your changes and commit them.
   ```
   git commit -m "Description of my changes"
   ```
6. **Test and Lint**: Make sure code passes all tests and linting.
    ```
    pytest tests/ -v
    ruff check .
    ```
7. **Push Your Changes**: Push your changes to your fork.
   ```
   git push origin my-new-branch
   ```
8. **Open a Pull Request**: At github, open a pull request with a description of your changes.


## Coding conventions :microscope:
When updating the code base, there are a few conventions I try to follow. The guiding principles are:

1. I try to write code (and documentation) that is simple and easy to read. If improving clarity means sacrificing a few milliseconds at runtime, that's fine.  
2. Don't reinvent wheels: if there is a useful function from another library (such as scikit learn's `classification_report()`) that is typically used to do something, let's show people how to use that tool in a demo. This is preferable to building a thin wrapper around the function. 

The specific coding conventions I follow:

1. **Docstrings**: We use NumPy-style docstrings.
2. **Linting**: We use `ruff` for linting (`ruff check .`).
3. **Testing**: All changes should pass tests in `pytest tests/ -v`.
4. **Imports**: Follow standard conventions (e.g., `import numpy as np`).

If unsure, just do your best to match the existing style, and I'll help fine-tune things in the PR! 

deepglue functions are currently divided into four utility modules (plotting, file, training, and runtime utilities). If you need help deciding where your code should go, feel free to ask. It's not always obvious.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

