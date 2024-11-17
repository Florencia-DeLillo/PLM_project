import esm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_utils import ProteinDataset, TaxonIdSampler, esm_compute, get_seq_rep, get_taxon_sequence_data, get_logits

BATCH_SIZE = 5
TSV_FILE = 'uniref_id_UniRef100_A0A003_OR_id_UniR_2024_11_17.tsv'
OUTPUT_DIR_REPS = "../demo/demo_results/"
OUTPUT_DIR_LOGI = ""
MODEL = esm.pretrained.esm2_t12_35M_UR50D()

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device: ", device)

collection = pd.read_csv(TSV_FILE, sep='\t')
dataset = ProteinDataset(collection)
sampler = TaxonIdSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: x, shuffle=False)

for batch in dataloader:

    sequences = [item['sequence'] for item in batch]
    names = [item['primary_accession'] for item in batch]

    print(sequences)
    print(names)

raise NotImplementedError

dataset_size = len(dataloader)

for n, batch in enumerate(dataloader):

    sequences = [item['sequence'] for item in batch]
    names = [item['primary_accession'] for item in batch]

    results, batch_lens, batch_labels, _ = esm_compute(seqs = sequences, 
                                                              names = names,
                                                              device=device,
                                                              model=MODEL)

    seq_reps = get_seq_rep(results, batch_lens)
    seq_logits = get_logits(results)

    for i, (resp, logits) in enumerate(zip(seq_reps, seq_logits)):
        torch.save(resp, f"{OUTPUT_DIR_REPS}{names[i]}_reps.pt")
        torch.save(resp, f"{OUTPUT_DIR_LOGI}{names[i]}_logi.pt")

    print(f"[{n+1/dataset_size*100:.4f}%]", "batch ", n+1, " saved.")