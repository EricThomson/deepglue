
# Contributing to deepglue

<img src="https://raw.githubusercontent.com/EricThomson/deepglue/main/docs/images/deep_glue_logo.png" alt="deepglue logo" align="right" width="125">

We are thrilled that you're interested in contributing to deepglue. :heart:  :hammer: 

There are many different ways you can contribute, from sharing ideas in discussions, issues, to helping improve documentation and the code base. Any help is welcome and appreciated! 

 The point of deepglue is to make deep learning projects easier, and we want to make *contributing* easier too, so please don't be shy about asking how it's done. :family:

## Ways to contribute

### Community
Partipate in the community, either in the issue queue or in discussion threads. If you find problems or bugs, or just have a question, feel free to ask.

### Documentation
Our documentation is always a work in progress. Any help keeping it clear and up-to-date is appreciated. This includes the demo notebooks.

### Code
- **New features**: if you have an idea for a new feature, please let us know! If you'd like to contribute the feature, we'll help you do it. At the very least, we'll add it to the [roadmap](https://github.com/EricThomson/deepglue/issues/1)! :world_map:
- **Code reviews**: if you have suggestions for how we can improve our code, please open a pull request and provide feedback! 
- **Bug fixes**: If you find a problem, a bug :bug:, please let us know by opening an issue or a bug fix (see below on how to do a pull request). 


## How to contribute code: making a pull request
Below is a brief summary of how to make a pull request (PR) to deepglue. Please keep in mind that we welcome first-time contributors: if you have never done this before, please reach out and we'll help you through the process. 

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

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

Thank you for being a part of deepglue! :heart:
