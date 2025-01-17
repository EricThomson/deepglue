# deepglue demos
This folder contains demo notebooks for using the deepglue package. 

## Available demos
- **fashion_mnist**: the base demo -- it shows how to use deepglue to set up a basic classification project, train a network, and do some common visualizations (and an interactive visualization of the features in the network).  

## To run locally
1. Clone the repo
2. Create a virtual environment and install pytorch (see below).
3. Navigate to the relevant demo folder (e.g., `cd demos/fashion_mnist`)
4. Install the requirements in the demo folder (`pip install -r requirements.txt`)
5. Open the notebook. E.g., `jupyter lab fashion_demo.ipynb`

### To install PyTorch
Pytorch works really hard to ensure things play well on all operating systems. I recommend go here: https://pytorch.org/get-started/locally/, select the values from their options and then copy the command where it says `Run this command`. To verify your install of torch is seeing the GPU, go into python and enter: `import torch, torch.cuda.is_available()`. It should return `True`.