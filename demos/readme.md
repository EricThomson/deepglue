# deepglue demos
Demo notebooks for the deepglue package. 
Home page for deepglue: https://github.com/EricThomson/deepglue

## Available demos
- **fashion_mnist**: the base demo, which shows how to use deepglue to set up a basic classification project, train a network, and evalute its performance. It demonstrates some of the key visualization, training, and file utilities. 

## To run a demo locally
1. Clone the repo
2. Create a virtual environment and install pytorch (see below).
3. Navigate to the relevant demo folder (e.g., `cd demos/fashion_mnist`)
4. Install the requirements in the demo folder (`pip install -r requirements.txt`)
5. Open the notebook. E.g., `jupyter lab fashion_demo.ipynb`

### To install PyTorch
The PyTorch team works hard to ensure things play well on all operating systems. I recommend go here: https://pytorch.org/get-started/locally/, select the values in the matrix of options you want, and then copy the command where it says `Run this command`. To verify your install of torch is seeing the GPU, go into python and enter: `import torch, torch.cuda.is_available()`. It will return `True` when things are working.