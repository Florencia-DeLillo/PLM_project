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
import pandas as pd
import multiprocessing
from token_mask import mask_single
from torch.nn.utils.rnn import pad_sequence
import warnings

warnings.filterwarnings('ignore')

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


def kernel_similarity_matrix(repr):
    """
    Calculates the cosine similarity between each pair of token embeddings on the kernel
    """
    if isinstance(repr, list):
        repr = torch.stack([torch.tensor(k) for k in repr])  # Convert list to tensor
    
    # If kernel is a PyTorch tensor, move it to CPU and convert to NumPy
    # if not isinstance(kernel, torch.Tensor):
    #     kernel = kernel.cpu().detach().numpy()  # Move to CPU and detach if needed
    
    #print(type(repr))  # Debugging print
    
    repr = torch.nn.functional.normalize(repr, p=2, dim=1)

    cosine_similarity_matrix = torch.mm(repr, repr.T)

    #print(cosine_similarity_matrix.shape)

    return cosine_similarity_matrix

def kernel_mse_alignment_loss(teacher_kernel, student_kernel):
    """
    Calculates the MSE kernel alignment loss between teacher and student
    """
    #print("zero")
    kernel_similarity_matrix(teacher_kernel)
    #print("zero")
    teacher_matrix = torch.tensor(kernel_similarity_matrix(teacher_kernel))
    #print("zero")
    student_matrix = torch.tensor(kernel_similarity_matrix(student_kernel))
    #print("zero")

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

        alignment_loss = kernel_mse_alignment_loss(teacher_rep, student_rep)

        logits_loss = logits_mse_loss(teacher_logits, student_logits)

        total_loss = self.weight_rep * alignment_loss + self.weight_logits * logits_loss

        return total_loss


def get_seq_rep(results, batch_lens, layer = 33):
    """
    Get sequence representations from esm_compute
    """
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

def load_teacher_results(result_type:str, batch_number: int, path ='../data/outputs/'):
    if result_type == 'logi':
        path = path + f'teacher_logi/batch_{batch_number+1}_logi.pt'

    elif result_type == 'reps':
        path = path + f'teacher_reps/batch_{batch_number+1}_reps.pt'
    else:
        raise ValueError('La cagaste mano')
    
    result = torch.load(path)

    return result

############################################################

#DATA LOADING
BATCH_SIZE = 8
CSV_FILE = '../data/raw/uniprot_data_500k_sampled_250.csv'
# OUTPUT_DIR_REPS = "../data/outputs/student_reps/"
# OUTPUT_DIR_LOGI = "../data/outputs/student_logi/"
#MODEL = esm.pretrained.esm2_t6_8M_UR50D()
REP_LAYER= 6 #ensure it matches the model
SEQ_MAX_LEN = 256

collection = pd.read_csv(CSV_FILE)
dataset = ProteinDataset(collection, SEQ_MAX_LEN)
sampler = TaxonIdSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: x, shuffle=False)

#PARAM SET
num_epochs = 1
learning_rate = 1e-4
weight_rep = 0.5
weight_logits = 0.5


#TRAINING LOOP



checkpoints = True
cp_dir = "checkpoints"
cp_freq = 200

# get data
#_, _, collection = connect_db()
#dataset = ProteinDataset(get_taxon_sequence_data(collection))

############################ TEMP FOR TESTING

# load models
student_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# initialize batch converter
batch_converter = alphabet.get_batch_converter()

# train only student
student_model.train()

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device: ", device)
student_model.to(device)

# define optimizer and loss
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
distillation_loss = DistillationLoss(weight_rep, weight_logits)


for epoch in range(num_epochs):

    for i, batch in enumerate(dataloader):

        # extract sequences and names from the batch
        sequences = [item['sequence'] for item in batch]
        names = [item['protein_id'] for item in batch]

        # prepare data for batch conversion
        if names is None:
            names = [f'seq{i}' for i in range(len(sequences))]
        data = list(zip(names, sequences))

        batch_seed = i*BATCH_SIZE

        with multiprocessing.Pool() as pool:
            masking = pool.starmap(mask_single, [(n, item, batch_seed) for n, item in enumerate(batch)]) 
        seqs, masked_pos = zip(*masking)

        data_mask = list(zip(names, seqs))

        # check datatype of sequences - str or biotite
        if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
            pass  # all elements are strings
        else:
            data = [(x[0], str(x[1])) for x in data]

        # convert data to batch tensors
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # convert masked data to batch tensors
        masked_batch_labels, masked_batch_strs, masked_batch_tokens = batch_converter(data_mask)
        masked_batch_lens = (masked_batch_tokens != alphabet.padding_idx).sum(1)
        masked_batch_tokens = masked_batch_tokens.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass - teacher
    
        teacher_logits = load_teacher_results('logi', i)
        teacher_reps = load_teacher_results('reps', i)

        # forward pass - student representations
        student_res = student_model(batch_tokens, repr_layers=[REP_LAYER], return_contacts=False)
        student_reps = get_seq_rep(student_res, batch_lens, layer=REP_LAYER)

        #forward pass - student logits
        
        student_masked_res = student_model(masked_batch_tokens, repr_layers=[REP_LAYER], return_contacts=False)
        student_logits = get_logits(student_masked_res)

        masked_logi = []
        for i, positions in enumerate(masked_pos):
            positions = [i+1 for i in positions] #account for <str> token
            masked_logi.append(student_logits[i, positions, :])
        # stack into a tensor with padding (seq have different number of masked pos)
        masked_student_logits = pad_sequence(masked_logi, batch_first=True, padding_value=0.0)

        # compute loss and backprop
        loss = distillation_loss(teacher_reps, teacher_logits, student_reps, masked_student_logits)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # tensorflow-like checkpoints
    # if checkpoints:
    #     if (epoch + 1) % cp_freq == 0:
    #         path = f'cp_epoch_{epoch+1}.pt'
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': student_model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss.item(),
    #         }, path)
    #         print(f'Checkpoint saved: {path}')