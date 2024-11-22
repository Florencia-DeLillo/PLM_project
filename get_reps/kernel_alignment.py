import torch
import torch.nn as nn
import numpy as np
import esm
import random
from torch.utils.data import DataLoader
from data_utils import ProteinDataset, TaxonIdSampler, get_seq_rep, get_logits
#For knowledge distillation
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as functional
import simsimd



def kernel_similarity_matrix(repr):
    """
    Calculates the cosine similarity between each pair of token embeddings on the kernel
    """
    if isinstance(repr, list):
        repr = torch.stack([torch.tensor(k) for k in repr])  # Convert list to tensor
    
    # If kernel is a PyTorch tensor, move it to CPU and convert to NumPy
    # if not isinstance(kernel, torch.Tensor):
    #     kernel = kernel.cpu().detach().numpy()  # Move to CPU and detach if needed
    
    print(type(repr))  # Debugging print
    
    repr = torch.nn.functional.normalize(repr, p=2, dim=1)

    cosine_similarity_matrix = torch.mm(repr, repr.T)

    print(cosine_similarity_matrix.shape)

    return cosine_similarity_matrix


vec1 = np.random.randn(1536).astype(np.float32)
vec2 = np.random.randn(1536).astype(np.float32)
dist = simsimd.cosine(vec1, vec2)

print(vec1)
print(vec2)
print(dist)

X = [[0, 0, 0], [1, 1, 1]]
Y = [[1, 0, 0], [1, 1, 0]]
print(X)
cosine_similarity(X, Y)