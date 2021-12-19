import os
import torch

import numpy as np

def save_model(model, epoch, path):
    print("Saving ...")
    if not os.path.isdir(path):
        os.mkdir(path)
    name = os.path.join(path, f"model_{epoch}.pth")
    torch.save(model.state_dict(), name)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def get_mean_dist(vectors1, vectors2):
    vector_norms = np.linalg.norm(vectors1 - vectors2, axis=1)

    return np.mean(vector_norms)

def read_from_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines
