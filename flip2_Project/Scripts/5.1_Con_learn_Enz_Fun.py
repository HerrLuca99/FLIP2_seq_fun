# %%
import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import math


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

# train_df = pd.read_csv(f'{DATA_DIR}/TB_family_merged_train.csv')
# test_df = pd.read_csv(f'{DATA_DIR}/test/csv_id/test_df/TB_family_merged_test.csv')
# df with ESM2 embeddings
df_pickle = pd.read_pickle(f'{DATA_DIR}/test/csv_id/pickel/esm2_embeddings_merged.pkl')
# df with preformed paires
df_paires = pd.read_csv(f'{DATA_DIR}/test/csv_id/paires/df_paires.csv') 

# %%
num_paires = df_paires['id_1'].count()
print(num_paires)

# %%
## This needs all be extended for the triplet loss at some point
## Build the Dataset class
class Pairdataset(Dataset):
    def __init__(self, pair_indices, embeddings):
        # self.main_store = df_pickle
        self.pair_indices = pair_indices
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, idx):
        row = self.pair_indices.iloc[idx]
        emb_a = torch.tensor(self.embeddings[row['id_1']], dtype = torch.float32)
        emb_b = torch.tensor(self.embeddings[row['id_2']], dtype = torch.float32)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return emb_a, emb_b, label

dataset = Pairdataset(df_paires, df_pickle)
print(len(dataset))
print(dataset[0])

# %%
# Data loader
batch_s = 64
loader = DataLoader(dataset, batch_size = batch_s, shuffle=True)

for emb_a, emb_b, label in loader:
    print(emb_a.shape, emb_b.shape, label.shape)
    break  # just check the first batch

# %%
# Model (Siamese network) Two identical neural networks where each parameter gets fet 
# into one, but weights are updated automatically for both  
in_dim = 1280
hid_dim = 512
out_dim = 128

class EnzProjectionHead(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()   # always call this first
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, p=2, dim=1)#p2 means L2 normalise along the embedding dimesion to reduce the embedding space itself
    
model = EnzProjectionHead(in_dim, hid_dim, out_dim)   
print(model)

# %%
# Contrastive loss function
# later on change the pos and neg weight depending on the class imbalance
# def pair_imbalance
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin

    def forward(self, z1, z2, labels, pos_weight=1.0, neg_weight=1.0):
        
        distance = F.pairwise_distance(z1, z2, p=2)

        pos_loss = pos_weight * (distance ** 2)
        neg_loss = neg_weight * F.relu(self.margin - distance)**2

        loss = torch.where(labels == 1, pos_loss, neg_loss)

        return loss.mean()

loss_model = ContrastiveLoss(margin=1.0)
print(loss_model)


# %%
#Training loop 
target_steps = 1000
learning_rate = 0.001
batch_size = 64


num_batches = math.ceil(len(dataset) / batch_size )
num_epochs  = math.ceil(target_steps / num_batches)


# %%
#Training loop 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = model.to(device)

for epoch in range(num_epochs):
    total_loss = 0

    for x1, x2, label in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)

        out1 = model(x1)
        out2 = model(x2)

        loss = loss_model(
            out1, 
            out2,
            label,
            0.5
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches:.4f}")


# %%
## Evaluate the model 
# Generate embeddings for all sequences
def get_con_model_embeddings(model, embedding_dict, family_prefix, device):
    model.eval()
    projected = []
    ids =[]
    batch_tensor = []
    batch_ids = []

    with torch.no_grad():
        for seq_id, raw_emb in tqdm(embedding_dict.items()):
            if family_prefix not in seq_id:
                continue
            batch_tensor.append(torch.tensor(raw_emb, dtype=torch.float32))
            batch_ids.append(seq_id)

            if len(batch_tensor) == batch_size:
                batch = torch.stack(batch_tensor).to(device)
                out = model(batch).cpu()
                projected.append(out)
                ids.extend(batch_ids)
                batch_tensor = []
                batch_ids = []

        if batch_tensor:
            batch = torch.stack(batch_tensor).to(device)
            out = model(batch).cpu()
            projected.append(out)
            ids.extend(batch_ids)


    return torch.cat(projected).numpy(), ids
#toch.stack for single sequences which are get stacked together vs cat for 2D tensor


# %%
# Check if it worked
proj_matrix, seq_id = get_con_model_embeddings(model, df_pickle, "TB", device)
print(proj_matrix.shape)
print(seq_id)


# %%
#Check Data availability before UMAP
print(proj_matrix.shape)
print(len(seq_id))
#Plot
df_csv_for_pick = pd.read_csv('/home/lherrmann/projects/enzyme-design/flip2_Project/test/csv_id/uni_scale/TB.csv')
print(len(df_csv_for_pick))

print(df_csv_for_pick.columns.tolist())

# %%
#For UMAP preselct the sequences otherwise need to long to be able to do it
# be cautious it currently only plots on thetraining data otherwise needs to long for 600 k sequences change later
unique_train_ids = set(df_paires['id_1']).union(set(df_paires['id_2']))
print(len(unique_train_ids))

id_to_scale = df_csv_for_pick.set_index('id')['uni_scale'].to_dict()
scales = [id_to_scale[sid] for sid in seq_id]

filtered_idx = [i for i, sid in enumerate(seq_id) if sid in unique_train_ids]
proj_sub = proj_matrix[filtered_idx]
scales_sub = [scales[i] for i in filtered_idx]

print(proj_sub)
print(scales_sub)

# %%
#UMAP: compresses 128 dimensional space into 2D space
import umap

reducer = umap.UMAP(n_components=2, random_state=42)# random state: every time you run this code, you get the exact same map
umap_results = reducer.fit_transform(proj_sub)
print(umap_results.shape)




# %%
plt.figure(figsize=(10,8))
scatter = plt.scatter(x = umap_results[:, 0], y = umap_results[:, 1], cmap = 'viridis', c=scales_sub, s=5 )

plt.colorbar(scatter, label='Uni_scale')
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.show()


# %%
Next steps:
1. Test loop
2. extending to negative paires and tripplet loss (anchor, neg and pos pair)
3. Hyperparameter tuning
4. increast Training data batch_size
5. Play around with pairing 
6. include hamming distance
7. 