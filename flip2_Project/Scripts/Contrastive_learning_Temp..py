# %%
# =============================================================================
# CONTRASTIVE LEARNING FOR ENZYME FUNCTION PREDICTION
# =============================================================================
# Goal: Train a neural network that learns a "embedding space" where enzymes
# with the same EC number (enzyme class) are close together, and enzymes with
# different EC numbers are pushed apart.
#
# The key idea of contrastive learning:
#   - Take two enzyme embeddings (numerical representations of protein sequences)
#   - If they have the same EC class → push them CLOSER in the learned space
#   - If they have different EC classes → push them FURTHER APART
#
# This is different from normal classification: instead of predicting a label
# directly, we learn a geometry of the space so we can later do nearest-neighbor
# classification (find the closest training example to a test example).
# =============================================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# --- PyTorch core imports ---
import torch                              # The main PyTorch library
from collections import defaultdict       # A dict that auto-creates missing keys
from tqdm import tqdm                     # Progress bars for loops
from torch.utils.data import Dataset      # Base class for custom datasets
import torch.nn as nn                     # Neural network building blocks (layers, loss functions)
import torch.nn.functional as F           # Stateless functions: relu, softmax, distance, etc.
import numpy as np
from torch.utils.data import DataLoader   # Efficiently batches and loads data during training


# %%
# --- Check hardware: GPU vs CPU ---
# PyTorch can run computations on the GPU (much faster) or fall back to CPU.
# CUDA is NVIDIA's GPU computing platform. If you don't have an NVIDIA GPU,
# CUDA will not be available and everything runs on CPU (slower but still works).
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %%
DATA_DIR = '/home/lherrmann/projects/enzyme-design/flip2_Project/'


# %%
# --- Load data ---
# train_df: table with columns 'Entry' (protein ID) and 'EC number' (enzyme class label)
# test_df:  same structure, used only for final evaluation (never seen during training)
# pickle_df: pre-computed ESM2 embeddings — each protein is represented as a 2560-dim vector
# paires_df: pre-built pairs of proteins with similarity labels (can be used instead of building pairs on the fly)
train_df = pd.read_csv(f'{DATA_DIR}/protein_train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/30_protein_test.csv')

# df with ESM2 embeddings
pickle_df = pd.read_pickle(f'{DATA_DIR}/test/csv_id/pickel/')

# df with preformed paires
paires_df = pd.read_csv(f'{DATA_DIR}/test/csv_id/paires/')


# %%
train_df.head()
test_df.head()
pickle_df.head()
paires_df.head()


# %%
# =============================================================================
# BUILD INDEX STRUCTURES
# =============================================================================
# We need fast lookups during training. Instead of working with protein IDs
# (strings) everywhere, we assign each protein an integer index i.
# Then we build several dictionaries:
#   idx_to_embedding[i]  → the ESM2 vector for protein i
#   idx_to_label[i]      → the EC class string for protein i  (e.g. "1.2.3")
#   value_to_index[entry]→ protein string ID → integer index
#   class_to_indices[ec] → list of all integer indices that belong to EC class ec
# =============================================================================

from collections import defaultdict

# Also do the same for proteins
idx_to_embedding = {}
value_to_index = {}
idx_to_label = {}
# Build a mapping: protein Entry ID → its ESM2 embedding vector
seq_to_embedding = dict(zip(df['Entry'].values, df['embedding'].values))

def build_idxs(ec_level):
    """
    Build lookup dictionaries for fast index-based access during training.

    ec_level: how many EC number digits to use as the class label.
              EC numbers look like "1.2.3.4" (4 levels of hierarchy).
              ec_level=1  → class is just "1"       (very broad)
              ec_level=4  → class is "1.2.3.4"      (very specific)
    """
    i = 0
    idx_to_embedding = {}   # integer index → ESM2 embedding vector (numpy array)
    value_to_index = {}     # protein Entry string → integer index
    idx_to_label = {}       # integer index → EC class string
    idx_to_entry = {}       # integer index → protein Entry string (for debugging)
    class_to_indicies = defaultdict(list)  # EC class string → list of integer indices

    for entry, ec in train_df[['Entry', 'EC number']].values:
        # Only include proteins that actually have a pre-computed embedding
        if seq_to_embedding.get(entry) is not None:
            # .flatten() turns a 2D array into a 1D vector (required by linear layers)
            clipped_embedding = seq_to_embedding.get(entry).flatten()

            # Store the flat embedding at index i
            idx_to_embedding[i] = clipped_embedding
            idx_to_entry[i] = entry
            value_to_index[entry] = i

            # Truncate the EC number to the requested level of hierarchy.
            # e.g. ec="1.2.3.4", ec_level=2  →  class_number="1.2"
            class_number = str('.'.join(ec.split('.')[:ec_level]))
            idx_to_label[i] = class_number

            # Remember which proteins belong to each EC class
            # (used later for balanced sampling)
            class_to_indicies[class_number].append(i)
            i += 1

    return idx_to_embedding, idx_to_label, value_to_index, class_to_indicies


def build_train_test_df(idx_to_embedding, idx_to_label, num_pairs):
    """
    Randomly sample `num_pairs` pairs of proteins and assign similarity labels.

    Returns:
        all_pair_embeddings: list of [i, j] index pairs
        all_labels:          list of labels:
                               +1  if protein i and j share the same EC class (similar)
                               -1  if they belong to different EC classes   (dissimilar)

    Note: this is the simplest possible strategy — pure random sampling.
    A smarter strategy would oversample hard negatives (proteins that are
    almost the same EC class but not quite).
    """
    all_pair_embeddings, all_labels = [], []
    num_idxs = len(idx_to_embedding) - 1

    for i in tqdm(range(0, num_pairs)):
        # Pick 2 random protein indices
        sample_i = random.sample(range(0, num_idxs), 2)
        all_pair_embeddings.append([sample_i[0], sample_i[1]])

        # Label: +1 if same class, -1 if different class
        if idx_to_label.get(sample_i[0]) == idx_to_label.get(sample_i[1]):
            all_labels.append(1)   # similar pair
        else:
            all_labels.append(-1)  # dissimilar pair

    return all_pair_embeddings, all_labels


# %%
# =============================================================================
# NEURAL NETWORK DEFINITION
# =============================================================================
# In PyTorch, you define a network by subclassing nn.Module.
# You must implement:
#   __init__  → define all the layers
#   forward   → define how data flows through those layers
#
# This is a simple 2-layer MLP (multi-layer perceptron / feedforward network):
#   Input embedding (2560-dim)
#   → Linear layer: hidden_dim
#   → LayerNorm + ReLU + Dropout
#   → Linear layer: latent_dim (output)
#
# The output is the "projected" embedding. The contrastive loss then operates
# on these projected embeddings, not the raw ESM2 embeddings.
# =============================================================================

class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.01):
        """
        input_dim:  size of the input embedding (2560 for ESM2)
        hidden_dim: size of the intermediate representation (1024)
        output_dim: size of the final output / latent embedding (512)
        dropout:    fraction of neurons randomly zeroed during training
                    to prevent overfitting (default 1%)
        """
        super(SimpleNetwork, self).__init__()
        # NOTE: protein_layer and reaction_layer are commented out.
        # They were intended for a dual-encoder design (one branch for protein,
        # one for reaction SMILES). Currently unused — the forward() will error
        # if input_type='protein' because protein_layer is not defined.
        # self.protein_layer = nn.Linear(input_dim, hidden_dim)
        # self.reaction_layer = nn.Linear(input_dim, hidden_dim)

        # nn.Linear(in, out): a fully-connected layer — multiplies input by a
        # weight matrix and adds a bias vector. Learnable parameters.
        self.hidden_layer = nn.Linear(hidden_dim, output_dim)

        # nn.Dropout: randomly sets some neuron activations to 0 during training.
        # This prevents the network from memorising training data (overfitting).
        self.dropout = nn.Dropout(dropout)

        # nn.LayerNorm: normalises each sample independently so activations
        # have mean≈0 and variance≈1. Helps training stability.
        self.norm = nn.LayerNorm(hidden_dim)

        # Activation functions (non-linearities):
        # ReLU(x) = max(0, x)  — zeroes out negative values, allows deep nets to learn
        self.relu = nn.ReLU()
        # Sigmoid(x) = 1/(1+e^-x) — squashes output to (0,1) — not used currently
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, input_type='protein'):
        """
        Defines the forward pass: how input `x` flows through the network.
        PyTorch automatically builds the computation graph here for backprop.

        x:          input tensor of shape (batch_size, input_dim)
        input_type: 'protein' or anything else — selects which first layer to use.
                    NOTE: this will currently crash because protein_layer /
                    reaction_layer are commented out above.
        """
        # Step 1: First linear layer (protein_layer or reaction_layer)
        if input_type == 'protein':
            x = self.protein_layer(x)   # shape: (batch, hidden_dim)
        else:
            x = self.reaction_layer(x)  # shape: (batch, hidden_dim)

        # Step 2: Normalise
        x = self.norm(x)

        # Step 3: Apply non-linearity
        x = self.relu(x)

        # Step 4: Second linear layer projects down to latent_dim
        x = self.hidden_layer(x)        # shape: (batch, output_dim)
        return x

# Note: x is the mean-pooled ESM2 embedding for the protein sequence


# =============================================================================
# CONTRASTIVE LOSS FUNCTION
# =============================================================================
# Standard contrastive loss (Hadsell et al. 2006):
#
#   For a SIMILAR pair   (label = +1):   loss = distance²
#       → we want distance to be small, so penalise large distances
#
#   For a DISSIMILAR pair (label = -1):  loss = max(0, margin - distance)²
#       → we want distance > margin, so only penalise if they're still too close
#
# margin: a hyperparameter that sets the "target minimum distance" for
#         dissimilar pairs. Typical value: 0.5–2.0.
# =============================================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, labels, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        """
        z1, z2:     (batch_size, latent_dim) — two batches of projected embeddings
        labels:     (batch_size,)            — +1 for similar, -1 for dissimilar
        margin:     minimum required distance for dissimilar pairs
        pos_weight: multiplier for positive-pair loss (used to balance class imbalance)
        neg_weight: multiplier for negative-pair loss
        """
        # F.pairwise_distance computes the L2 (Euclidean) distance between
        # corresponding rows of z1 and z2: ||z1[i] - z2[i]||₂
        # Result shape: (batch_size,)
        distances = F.pairwise_distance(z1, z2, p=2)

        # Similar pairs: minimise squared distance (pull them together)
        pos_loss = pos_weight * (distances ** 2)

        # Dissimilar pairs: if distance < margin, push them further apart.
        # F.relu(margin - distance) is 0 when distance >= margin (no penalty),
        # positive when distance < margin (pairs are too close).
        neg_loss = neg_weight * F.relu(margin - distances) ** 2

        # torch.where(condition, value_if_true, value_if_false)
        # Select pos_loss for similar pairs, neg_loss for dissimilar pairs
        loss = torch.where(labels == 1, pos_loss, neg_loss)

        # Average loss over the batch
        return loss.mean()


# =============================================================================
# DATASET CLASSES
# =============================================================================
# PyTorch's Dataset abstraction requires implementing:
#   __len__     → how many samples in the dataset
#   __getitem__ → how to fetch sample at position idx
# DataLoader wraps a Dataset to automatically:
#   - shuffle data each epoch
#   - batch samples together
#   - load batches in parallel (num_workers)
# =============================================================================

class PointerPairedDataset(Dataset):
    """
    Memory-efficient dataset for contrastive learning.

    Instead of storing all embedding pairs directly (which would duplicate data),
    we store just the *indices* of pairs and look up the actual embeddings on demand.

    main_store:    dict {int → embedding tensor}  — one embedding per protein
    pair_indices:  list of [i, j] pairs           — indices into main_store
    labels:        list of +1/-1 labels for each pair
    """
    def __init__(self, main_store, pair_indices, labels):
        self.main_store = main_store     # the full embedding dictionary
        self.pair_indices = pair_indices # list of index pairs, e.g. [[0, 5], [2, 8], ...]
        self.labels = labels             # list of labels, e.g. [1, -1, 1, ...]

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        # DataLoader calls this with idx = 0, 1, 2, ... to build a batch
        idx1, idx2 = self.pair_indices[idx]
        item1 = self.main_store[idx1]  # look up first protein's embedding
        item2 = self.main_store[idx2]  # look up second protein's embedding
        label = self.labels[idx]
        return item1, item2, label     # DataLoader stacks these into tensors of shape (batch, dim)


class PairedEmbeddingsDataset(Dataset):
    """
    Simpler alternative: stores the actual embedding tensors for all pairs directly.
    Easier to use but uses more memory (embedding pairs are stored twice if a
    protein appears in many pairs).

    tensor1, tensor2: numpy arrays of shape (num_pairs, embedding_dim)
    labels:           array of shape (num_pairs,)
    """
    def __init__(self, tensor1, tensor2, labels):
        assert tensor1.shape[0] == len(labels), "The number of labels must match the number of rows in the tensors"
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # torch.tensor(..., dtype=torch.float32): convert numpy → PyTorch tensor.
        # float32 is the standard floating-point type for neural network computation.
        return torch.tensor(self.tensor1[idx], dtype=torch.float32), \
               torch.tensor(self.tensor2[idx], dtype=torch.float32), \
               self.labels[idx]


class BasicDataset(Dataset):
    """
    Generic dataset for (data, label) pairs.
    Used at evaluation time: each sample is one protein embedding + its EC label.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


# =============================================================================
# EVALUATION HELPERS
# =============================================================================
# After training, we do NOT add a classification head. Instead we use
# nearest-neighbour (kNN with k=1) classification:
#   1. Pass all training proteins through the trained network → train embeddings
#   2. For each test protein → compute its embedding
#   3. Find the closest training embedding (by Euclidean distance)
#   4. Predict the label of that nearest neighbour
#
# This works because the contrastive loss has trained the space so that
# same-class proteins are close together.
# =============================================================================

def compute_all_train_embeddings(model, train_dataset):
    """
    Run every training sample through the model to get its projected embedding.
    These are cached so we don't have to recompute them for every test sample.

    model.eval():      switches Dropout off (we want deterministic embeddings)
    torch.no_grad():   disables gradient tracking — saves memory, speeds up inference
    .unsqueeze(0):     adds a batch dimension: shape (dim,) → (1, dim)
                       because the model expects (batch_size, dim) inputs
    """
    embeddings = []
    labels = []
    model.eval()  # evaluation mode: dropout is disabled

    with torch.no_grad():  # no gradient computation needed during inference
        for x, y in tqdm(train_dataset):
            emb = model.forward(x.unsqueeze(0))  # (1, latent_dim)
            embeddings.append(emb)
            labels.append(y)

    return embeddings, labels


def classify_with_softmax(train_embs, train_labels, test_emb, num_classes):
    """
    Soft nearest-neighbour classification using softmax over negative distances.

    Instead of hard kNN (just pick the closest), this converts all distances
    to probabilities and aggregates them class-by-class.

    distances = ||train_emb - test_emb||₂  for each training sample
    similarity = -distance  (close = high similarity)
    probs = softmax(similarity)  → sum to 1, higher for closer points

    Then sum probabilities for each class → class with highest sum wins.
    """
    distances = torch.norm(train_embs - test_emb, dim=1)
    similarities = -distances           # negate: closer → higher value
    probs = F.softmax(similarities, dim=0)  # turn similarities into probabilities

    # Accumulate probability mass per class
    class_probs = torch.zeros(num_classes)
    for prob, label in zip(probs, train_labels):
        class_probs[label] += prob

    return class_probs / class_probs.sum()  # renormalise (should already sum to ~1)


def classify_by_nearest(train_embs, train_labels, test_emb, top_k=1):
    """
    Hard nearest-neighbour classification.

    Compute Euclidean distance from test_emb to every training embedding.
    Return the label of the single closest training point (top_k=1),
    or do majority voting among the top_k closest (kNN).

    torch.topk(-distances, k): gets the k smallest distances
    (we negate because topk returns the k LARGEST values)
    """
    distances = torch.norm(train_embs - test_emb, dim=1)  # shape: (num_train,)
    topk_indices = torch.topk(-distances, k=top_k).indices

    topk_labels = train_labels[topk_indices]
    print(topk_indices[:2])

    if top_k == 1:
        print(topk_labels[0])
        return topk_labels[0].item()  # .item() converts a 1-element tensor → Python scalar

    # Majority vote: return the most common label among the top-k neighbours
    predicted_label = torch.mode(topk_labels).values.item()
    return predicted_label


def evaluate_nearest_neighbor(model, train_dataset, test_dataset):
    """
    Full evaluation loop: compute accuracy using 1-nearest-neighbour classification.
    """
    train_embs, train_labels = compute_all_train_embeddings(model, train_dataset)
    print(train_labels[0])
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_true in test_dataset:
            # Get the projected embedding for this test protein
            test_emb = model.forward(x.unsqueeze(0)).squeeze(0)  # .squeeze(0) removes the batch dim
            y_pred = classify_by_nearest(train_embs, train_labels, test_emb)
            print(y_pred, y_true)
            correct += (y_pred == y_true)
            total += 1

    acc = correct / total
    print(acc)
    return acc


# %%
# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# These control the training process. You tune these to improve performance.
# =============================================================================

num_pairs  = 1000000  # total number of (protein1, protein2) training pairs
                      # more pairs → better coverage of the space → better model
batch_size = 1000     # how many pairs to process at once before updating weights
                      # larger = more stable gradients, but more GPU memory needed
input_dim  = 2560     # ESM2 embedding dimension (fixed by the pre-trained model)
hidden_dim = 1024     # size of the first hidden layer
hidden_dim_2 = 256    # (unused in current architecture)
latent_dim = 512      # size of the output / projected embedding
lr         = 0.001    # learning rate: step size for gradient descent
                      # too large → unstable training; too small → slow convergence
epochs     = 5        # how many times to loop over all pairs


# =============================================================================
# TRAINING LOOP
# =============================================================================
# This trains `num_models` independent models (ensemble).
# Each model sees a freshly sampled set of pairs.
# =============================================================================

num_models = 1
models = []

for model_i in range(0, num_models):
    # --- Build training pairs ---
    all_pair_embeddings, all_labels = build_train_test_df(idx_to_embedding, idx_to_label, num_pairs)

    # --- Create dataset and DataLoader ---
    # PointerPairedDataset stores index pairs; DataLoader batches them
    # num_workers=20: use 20 CPU threads to load data in parallel
    dataset = PointerPairedDataset(idx_to_embedding, all_pair_embeddings, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=20)

    # --- Select compute device ---
    # Move the model to GPU if available, otherwise stay on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cl_model = SimpleNetwork(input_dim, hidden_dim, latent_dim).to(device)  # .to(device) moves all weights to GPU

    # --- Loss function ---
    # ContrastiveLoss: pulls similar pairs together, pushes dissimilar pairs apart
    criterion = ContrastiveLoss()

    # --- Optimiser ---
    # Adam: an adaptive learning-rate gradient descent algorithm.
    # cl_model.parameters() passes all learnable weights to the optimiser.
    optimizer = torch.optim.Adam(cl_model.parameters(), lr=lr)

    # --- Compute class weights to handle imbalanced pairs ---
    # If there are far more negative pairs than positive, the loss would be dominated
    # by negatives. We scale up the rarer class to balance contributions.
    all_labels = np.array(all_labels)
    num_pos = (all_labels == 1).sum().item()    # count of similar pairs
    num_neg = (all_labels == -1).sum().item()   # count of dissimilar pairs
    total = num_pos + num_neg
    pos_weight = total / (num_pos)              # > 1 if positives are rare
    neg_weight = total / (num_neg)              # > 1 if negatives are rare

    # --- Epoch loop ---
    for epoch in range(epochs):
        # DataLoader yields batches of (x1, x2, label), each of shape (batch_size, dim)
        for x1, x2, label in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            # Move batch tensors to the same device as the model
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            # Forward pass: run both proteins through the network
            out1 = cl_model(x1)  # shape: (batch_size, latent_dim)
            out2 = cl_model(x2)  # shape: (batch_size, latent_dim)

            # Compute contrastive loss
            # Multiplied by 10000 to make the loss value easier to read/monitor
            # (the raw loss can be very small due to the embedding scale)
            loss = 10000 * criterion(
                out1.to('cuda').squeeze(),
                out2.to('cuda').squeeze(),
                label.to('cuda'),
                0.5,         # margin: dissimilar pairs must be > 0.5 apart
                pos_weight,
                neg_weight
            )

            # Backward pass:
            # 1. Zero out gradients from the previous step (PyTorch accumulates them by default)
            optimizer.zero_grad()
            # 2. Compute gradients of the loss w.r.t. all model parameters (backpropagation)
            loss.backward()
            # 3. Update model parameters: w = w - lr * gradient
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    models.append(cl_model)


# %%
# =============================================================================
# PREPARE DATA FOR EVALUATION
# =============================================================================
# Build BasicDataset objects for train and test — one embedding per protein
# (not pairs). These are used for nearest-neighbour evaluation.
# =============================================================================

uda = True  # NOTE: variable name looks like a typo (should probably be `cuda`)
DEVICE = torch.device("cuda" if cuda else "cpu")  # NOTE: `cuda` is undefined here — likely a bug

# --- Build training evaluation dataset ---
data, labels = [], []
for i, (entry, ec) in enumerate(train_df[['Entry', 'EC number']].values):
    if seq_to_embedding.get(entry) is not None:
        clipped_embedding = seq_to_embedding.get(entry).flatten()
        data.append(clipped_embedding)
        # Convert EC number string "1.2.3.4" → integer array [1, 2, 3, 4]
        # 'n' is replaced with '' to handle non-numeric EC digits (e.g. "1.n.3.4")
        ec = np.array([int(e.replace('n', '')) for e in ec.split('.')])
        labels.append(ec)

# torch.tensor(...): convert list of numpy arrays to a single PyTorch tensor
# .to(DEVICE): move to GPU/CPU
training_data   = torch.tensor(np.array(data)).to(DEVICE)
training_labels = torch.tensor(np.array(labels)).to(DEVICE)
train_dataset   = BasicDataset(training_data, training_labels)

# --- Build test evaluation dataset (same structure) ---
data, labels = [], []
for i, (entry, ec) in enumerate(test_df[['Entry', 'EC number']].values):
    if seq_to_embedding.get(entry) is not None:
        clipped_embedding = seq_to_embedding.get(entry).flatten()
        data.append(clipped_embedding)
        ec = np.array([int(e.replace('n', '')) for e in ec.split('.')])
        labels.append(ec)

test_data    = torch.tensor(np.array(data)).to(DEVICE)
test_labels  = torch.tensor(np.array(labels)).to(DEVICE)
test_dataset = BasicDataset(test_data, test_labels)

# DataLoaders for train (shuffled, large batches) and test (ordered, one-by-one)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)


# %%
# =============================================================================
# EVALUATION: NEAREST-NEIGHBOUR ACCURACY PER EC LEVEL
# =============================================================================
# EC numbers have 4 hierarchical levels: "1.2.3.4"
#   Level 1: enzyme class (very broad, 7 classes total)
#   Level 2: subclass
#   Level 3: sub-subclass
#   Level 4: specific reaction (most specific)
#
# We measure accuracy at each level independently.
# Getting level 4 right implies levels 1–3 are also right.
# =============================================================================

for model in models:
    print('--------------- Evaluating model ---------------------')

    # torch.no_grad(): no gradient computation — saves memory and speeds up evaluation
    with torch.no_grad():
        # Cache all training embeddings (projected through the trained model)
        train_embs, train_labels = compute_all_train_embeddings(model, train_dataset)

        correct_level4, correct_level3, correct_level2, correct_level1 = 0, 0, 0, 0
        total = 0

        # Stack list of (1, latent_dim) tensors → (num_train, latent_dim) matrix
        # .squeeze(1): remove the size-1 batch dimension
        # .cpu().numpy(): move to CPU and convert to numpy for np.linalg.norm
        train_embs_np  = torch.stack(train_embs).cpu().numpy().squeeze(1)
        train_labels_np = train_labels

        for x, y_true in tqdm(test_dataset):
            # Get projected embedding for this test protein
            test_emb = model.forward(x.unsqueeze(0)).squeeze(0).cpu().numpy()

            # np.linalg.norm(..., axis=1): Euclidean distance from test_emb to each row
            distances = np.linalg.norm(train_embs_np - test_emb, axis=1)  # shape: (num_train,)

            # Find the single closest training embedding
            topk_idx = np.argmin(distances)

            # Convert predicted and true labels from tensor → list of strings
            y_pred = [str(int(x)) for x in list(train_labels_np[topk_idx].cpu().numpy())]
            y_true = [str(int(x)) for x in list(y_true.cpu().numpy())]

            # Evaluate at each EC level:
            # Level 1: just the first digit, e.g. "1"
            correct_level1 += 1 if y_pred[0] == y_true[0] else 0
            # Level 2: first two digits joined, e.g. "12"
            correct_level2 += 1 if ''.join(y_pred[:2]) == ''.join(y_true[:2]) else 0
            # Level 3: first three digits, e.g. "123"
            correct_level3 += 1 if ''.join(y_pred[:3]) == ''.join(y_true[:3]) else 0
            # Level 4: all four digits (full EC number), e.g. "1234"
            correct_level4 += 1 if ''.join(y_pred) == ''.join(y_true) else 0
            total += 1


# %%
