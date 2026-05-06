#!/bin/bash
#SBATCH --job-name=esm2_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

mkdir -p logs

source ~/flip2/.venv/bin/activate

python ~/flip2/flip2_Project/Scripts/3.1_run_esm2_embeddings.py \
    --csv    /mnt/nfs/vol8t/home/lherrmann/flip2/Data/Input/merged.csv \
    --output /mnt/nfs/vol8t/home/lherrmann/flip2/Data/Output/esm2_embeddings_merged.pkl \
    --batch  20