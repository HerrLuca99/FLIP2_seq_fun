import argparse
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

parser = argparse.ArgumentParser()
parser.add_argument("--csv",    required=True,  help="Input CSV with 'id' and 'sequence' columns")
parser.add_argument("--output", required=True,  help="Output pickle path")
parser.add_argument("--batch",  type=int, default=20, help="Batch size (default: 20)")
args = parser.parse_args()

csv_path    = args.csv
output_path = args.output
batch_size  = args.batch

model_name = "facebook/esm2_t33_650M_UR50D"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name).to(device)
model.eval()

embeddings_dict = {}

with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):

        # Step 1: slice
        batch_df = df.iloc[i : i + batch_size]
        ids  = batch_df['id'].tolist()
        seqs = batch_df['sequence'].tolist()

        # Step 2: tokenize the list of sequences
        inputs = tokenizer(seqs, return_tensors="pt",
                            truncation=True, max_length=1024,
                            padding=True).to(device)

        # Step 3: forward pass
        outputs = model(**inputs)

        # Step 4: masked mean
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked = outputs.last_hidden_state * mask
        embedding = masked.sum(dim=1) / mask.sum(dim=1)

        # Step 5: store each sequence's embedding
        for j, pid in enumerate(ids):
            embeddings_dict[pid] = embedding[j].cpu().numpy()



    
  
# 4. Save to temporary folder/file
with open(output_path, "wb") as f:
    pickle.dump(embeddings_dict, f)

print(f"Success! Saved {len(embeddings_dict)} embeddings to {output_path}")
