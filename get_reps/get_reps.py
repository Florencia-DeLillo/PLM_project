import esm
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from data_utils import ProteinDataset, TaxonIdSampler, get_seq_rep, get_logits
import multiprocessing
from token_mask import mask_single
from torch.nn.utils.rnn import pad_sequence

# BATCH_SIZE = 8
# SEQ_MAX_LEN = 256
# #print(BATCH_SIZE)
# CSV_FILE = '../data/raw/uniprot_data_500k_sampled_250.csv'
# OUTPUT_DIR = "../data/outputs/teacher_reps/"
# MODEL = esm.pretrained.esm2_t33_650M_UR50D()
# REP_LAYER= 33 #ensure it matches the model
# TYPE = "reps" # reps or logi

BATCH_SIZE = 8
SEQ_MAX_LEN = 256
CSV_FILE = '../data/raw/uniprot_data_500k_sampled_250.csv'
OUTPUT_DIR = "../data/outputs/teacher_logi/"
MODEL = esm.pretrained.esm2_t33_650M_UR50D()
REP_LAYER= 33 #ensure it matches the model
TYPE = "logi" # reps or logi

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device: ", device)

collection = pd.read_csv(CSV_FILE)
dataset = ProteinDataset(collection, SEQ_MAX_LEN)
sampler = TaxonIdSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: x, shuffle=False)

#batch = dataloader[0]
#sequences = [item['sequence'] for item in batch]
#names = [item['taxon_id'] for item in batch]

#print(sequences)
#print(names)

#raise NotImplementedError

dataset_size = len(dataloader)
print(dataset_size)
for n, batch in enumerate(dataloader):

    if TYPE == "reps":
        seqs = [item['sequence'] for item in batch]
        #print(len(seqs[0]))
    elif TYPE == "logi":
        # perform masking
        batch_seed = n*BATCH_SIZE
        with multiprocessing.Pool() as pool:
            masking = pool.starmap(mask_single, [(i, item, batch_seed) for i, item in enumerate(batch)]) 
        seqs, masked_pos = zip(*masking)
    else: 
        raise KeyError
    
    names = [item['protein_id'] for item in batch]

    data = list(zip(names, seqs))
    
    model, alphabet = MODEL
    model.eval()
    model.to(device)

    batch_converter = alphabet.get_batch_converter()
 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # get results
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[REP_LAYER], return_contacts=True)

    if TYPE == "reps":
        res = get_seq_rep(results, batch_lens, layers=REP_LAYER)
    elif TYPE == "logi":
        res = get_logits(results)
        # trim logits into just masking positions
        masked_logi = []
        for i, positions in enumerate(masked_pos):
            positions = [i+1 for i in positions] #account for <str> token
            masked_logi.append(res[i, positions, :])
        # stack into a tensor with padding (seq have different number of masked pos)
        res = pad_sequence(masked_logi, batch_first=True, padding_value=0.0)
    else:
        raise KeyError

    # save the tensor as whole batch
    torch.save(res, os.path.join(OUTPUT_DIR, f"batch_{n+1}_{TYPE}.pt"))
    
    #print(f"[{(n+1)/dataset_size*100:.2f}%]", "batch ", n+1, " saved.")
    torch.cuda.empty_cache()