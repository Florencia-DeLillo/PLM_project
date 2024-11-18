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

mse_loss = nn.MSELoss()

def pad_to_match(teacher_kernel, student_kernel):
    """
    Just a precaution function. It assures that tokens embeddings in both teacher and student
    representations have the same shape. This will apply zero-padding the kernel with less dimensions.
    """
    rows = max(teacher_kernel.shape[0], student_kernel.shape[0])
    cols = max(teacher_kernel.shape[1], student_kernel.shape[1])
    new_teacher_kernel = functional.pad(teacher_kernel, (0, cols - teacher_kernel.shape[1], 
                                                            0, rows - teacher_kernel.shape[0]))
    new_student_kernel = functional.pad(student_kernel, (0, cols - student_kernel.shape[1], 
                                                            0, rows - student_kernel.shape[0]))
    return new_teacher_kernel, new_student_kernel


def kernel_similarity_matrix(kernel):
    """
    Calculates the cosine similarity between each pair of token embeddings on the kernel
    """
    print(type(kernel))
    if isinstance(kernel, list):
        kernel = torch.stack([torch.tensor(k) for k in kernel])  # Convert list to tensor
    
    # If kernel is a PyTorch tensor, move it to CPU and convert to NumPy
    if isinstance(kernel, torch.Tensor):
        kernel = kernel.cpu().detach().numpy()  # Move to CPU and detach if needed
    
    print(type(kernel))  # Debugging print

    
    return cosine_similarity(kernel)

def kernel_mse_alignment_loss(teacher_kernel, student_kernel):
    """
    Calculates the MSE kernel alignment loss between teacher and student
    """
    print("zero")
    kernel_similarity_matrix(teacher_kernel)
    print("zero")
    teacher_matrix = torch.tensor(kernel_similarity_matrix(teacher_kernel))
    print("zero")
    student_matrix = torch.tensor(kernel_similarity_matrix(student_kernel))
    print("zero")

    if teacher_matrix.shape != student_matrix.shape:
        teacher_matrix, student_matrix = pad_to_match(teacher_matrix, student_matrix)

    return mse_loss(teacher_matrix, student_matrix)

def logits_mse_loss(teacher_logits, student_logits):
    """
    Calculates the MSE loss between teacher and student logits
    """
    return mse_loss(teacher_logits, student_logits)


class DistillationLoss(nn.Module):
    def __init__(self, weight_rep=1.0, weight_logits=1.0):
        super(DistillationLoss, self).__init__()
        self.weight_rep = weight_rep
        self.weight_logits = weight_logits

    def forward(self, teacher_rep, teacher_logits, student_rep, student_logits):
        print("one")

        alignment_loss = kernel_mse_alignment_loss(teacher_rep, student_rep)
        print("one")
        logits_loss = logits_mse_loss(teacher_logits, student_logits)
        print("one")
        return torch.tensor(self.weight_rep * alignment_loss + self.weight_logits * logits_loss).cuda()


def get_seq_rep(results, batch_lens, layer = 33):
    """
    Get sequence representations from esm_compute
    """
    print(results["representations"])
    token_representations = results["representations"][layer]
 
    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
 
    return sequence_representations
 

def get_logits(results):
    """
    Get logits from esm_compute
    """
    logits = results["logits"]
    return logits







batch_size = 1
num_epochs = 1
learning_rate = 1e-4
weight_rep = 0.5
weight_logits = 0.5

checkpoints = True
cp_dir = "checkpoints"
cp_freq = 200

# get data
#_, _, collection = connect_db()
#dataset = ProteinDataset(get_taxon_sequence_data(collection))

############################ TEMP FOR TESTING
import json
import pandas as pd

def get_taxon_sequence_data2(collection):

    documents = collection['results']

    minimal_documents = [] # Initialize new empty dictionary

    for doc in documents:
    # Create a new dictionary with only the desired properties
        new_obj = {
            "primaryAccession": doc.get("primaryAccession",{}),
            "taxonId": doc.get("organism", {}).get("taxonId"),
            "value": doc.get("sequence", {}).get("value"),
            "length": doc.get("sequence", {}).get("length")
        }
        minimal_documents.append(new_obj)
    return minimal_documents

with open('../data/processed/uniref100_test.json', 'r') as file :
    json_data = json.load(file)
collection = pd.read_json(json.dumps(json_data))
dataset = ProteinDataset(get_taxon_sequence_data2(collection))
############################################

sampler = TaxonIdSampler(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dict_collate_fn)

# load models
teacher_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
student_model, _ = esm.pretrained.esm2_t6_8M_UR50D()

# initialize batch converter
batch_converter = alphabet.get_batch_converter()

# train only student
teacher_model.eval()
student_model.train()

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device: ", device)
teacher_model.to(device)
student_model.to(device)

# define optimizer and loss
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
distillation_loss = DistillationLoss(weight_rep=1.0, weight_logits=1.0)








    # training loop
for epoch in range(num_epochs):

    for batch in dataloader:

        # extract sequences and names from the batch
        sequences = [item['sequence'] for item in batch]
        names = [item['primary_accession'] for item in batch]

        # prepare data for batch conversion
        if names is None:
            names = [f'seq{i}' for i in range(len(sequences))]
        data = list(zip(names, sequences))

        # check datatype of sequences - str or biotite
        if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
            pass  # all elements are strings
        else:
            data = [(x[0], str(x[1])) for x in data]

        # convert data to batch tensors
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass - teacher
        with torch.no_grad():
            teacher_res = teacher_model(batch_tokens, repr_layers=[33], return_contacts=False)
            teacher_logits = get_logits(teacher_res)
            teacher_reps = get_seq_rep(teacher_res, batch_lens)

        # forward pass - student
        student_res = student_model(batch_tokens, repr_layers=[6], return_contacts=False)
        student_logits = get_logits(student_res)
        student_reps = get_seq_rep(student_res, batch_lens, layer=6)

        # compute loss and backprop
        loss = distillation_loss(teacher_reps, teacher_logits, student_reps, student_logits)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # tensorflow-like checkpoints
    if checkpoints:
        if (epoch + 1) % cp_freq == 0:
            path = f'cp_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, path)
            print(f'Checkpoint saved: {path}')