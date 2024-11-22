# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Data utilities and custom functions
from data_utils import ProteinDataset, TaxonIdSampler, get_seq_rep, get_logits
from token_mask import mask_single

# Model and Machine learning - metrics
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import esm


# Data processing and utilities
import numpy as np
import pandas as pd
import random
import multiprocessing
from tqdm import tqdm

# Warnings
import warnings

#Imports for loss

from loss_functions import *

warnings.filterwarnings('ignore')


def load_teacher_results(result_type:str, batch_number: int, path ='../data/outputs/'):
    if result_type == 'logi':
        path = path + f'teacher_logi/batch_{batch_number+1}_logi.pt'

    elif result_type == 'reps':
        path = path + f'teacher_reps/batch_{batch_number+1}_reps.pt'
    else:
        raise ValueError('Value error: expecting reps or logi string')
    
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
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#PARAM SET
num_epochs = 1
learning_rate = 1e-4
weight_rep = 0.5
weight_logits = 0.5
mlflow.set_experiment("Test_experiment")

#Parameters
with mlflow.start_run():

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs",num_epochs)
    mlflow.log_param("weight_rep",weight_rep)
    mlflow.log_param("weight_logits",weight_logits)

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



for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
    cumulative_loss=0.0
    for i, batch in enumerate(dataloader):
        
        
        # extract sequences and names from the batch
        sequences = [item['sequence'] for item in batch]
        names = [item['protein_id'] for item in batch]
        #..dont think they may sense since we are calculating metrics per epoch
        taxon_ids=set([item['taxon_id'] for item in batch])
        seq_lengths=[item['sequence_length'] for item in batch]
        mlflow.log_param("taxon_ids",taxon_ids)
        mlflow.log_param("avg_seq_length",sum(seq_lengths)/len(seq_lengths))

        


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
        cumulative_loss += loss.item()

    # Compute average metrics for the epoch
    avg_loss = cumulative_loss / len(dataloader)

    # Log metrics to MLflow
    mlflow.log_metric("avg_loss", avg_loss, step=epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

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