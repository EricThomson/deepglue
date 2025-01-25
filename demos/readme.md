# deepglue demos
Demo notebooks for the deepglue package. 
Home page for deepglue: https://github.com/EricThomson/deepglue

## Available demos
- **fashion_mnist**: the base demo, which shows how to use deepglue to set up a classification project, train/evaluate a network, and visualize how features cluster in its hidden layers.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EricThomson/deepglue/blob/main/demos/fashion_mnist/fashion_demo.ipynb)

## To run in Colab
Click on the `Open in Colab` button next to the demo. Demos will run *very* slowly if you don't have GPU enabled, so go to `Runtime` -> `Change runtime type` to make sure you have `T4 GPU` enabled. You can also click `View resources` to see how much RAM, GPU, Disk space you are using at any given time. With the free version of Colab, you aren't guaranteed GPU access. Colab Pro  makes GPU access much more reliable, but does cost $10 a month. If this is a problem, we recommend a local setup. Training with CPU only is very slow right now (we will add workarounds for this eventually). 

## To run locally
Running demos locally is typically much faster and gives more flexiblity than running in Colab. To run locally:
1. Clone the repo.
2. Create a virtual environment and install pytorch (see below).
3. Navigate to the relevant demo folder (e.g., `cd demos/fashion_mnist`)
4. Install dependencies (`pip install -r requirements.txt`)
5. Run the demo. E.g., `jupyter lab fashion_demo.ipynb`

### To install PyTorch
The PyTorch team works hard to ensure things play well on all operating systems. I recommend go here: https://pytorch.org/get-started/locally/, select the values you want from the matrix of options, and then copy the command where it says `Run this command`. To verify your install of torch is seeing the GPU, go into python and enter: `import torch, torch.cuda.is_available()`. It will return `True` when things are working.

### If you run into problems
If you need help, please start a [Discussion](https://github.com/EricThomson/deepglue/discussions). If you find a bug please [open an issue](https://github.com/EricThomson/deepglue/issues).

### Future demos planned
- Imbalanced data 
- Object detection
- Object segmentation
- :sparkles: :sparkles: :sparkles: 