# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
import torch                              # The main PyTorch library
from collections import defaultdict       # A dict that auto-creates missing keys
from tqdm import tqdm                     # Progress bars for loops
from torch.utils.data import Dataset      # Base class for custom datasets
import torch.nn as nn                     # Neural network building blocks (layers, loss functions)
import torch.nn.functional as F           # Stateless functions: relu, softmax, distance, etc.
import numpy as np
from torch.utils.data import DataLoader



# %%
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %%
DATA_DIR = '/home/lherrmann/projects/enzyme-design/flip2_Project/'

train_df = pd.read_csv(f'{DATA_DIR}/protein_train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/30_protein_test.csv')
# df with ESM2 embeddings
pickle_df = pd.read_pickle(f'{DATA_DIR}/test/csv_id/pickel/')
# df with preformed paires
paires_df = pd.read_csv(f'{DATA_DIR}/test/csv_id/paires/')



# %%




