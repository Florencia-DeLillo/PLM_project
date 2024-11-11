import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
from functools import reduce
from collections import defaultdict
import os
from typing import Union
from pathlib import Path
import esm

def esm_compute(seqs: list, names: list=None, model: Union[str, torch.nn.Module]="esm1v", rep_layer: int=33, device=None):
    """
    Compute the of esm_tools models for a list of sequences.
 
    Args:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence.
        names (list, default None): list of names/labels for protein sequences.
            If None sequences will be named seq1, seq2, ...
        model (str, torch.nn.Module): choose either esm2, esm1v or a pretrained model object.
        rep_layer (int): choose representation layer. Default 33.
        device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'.
 
    Returns: representations (list) of sequence representation, batch lens and batch labels
 
    Example:
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        results, batch_lens, batch_labels = esm_compute(seqs)
    """
    # detect device
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
 
    # on M1 if mps available
    #if device == torch.device(type='cpu'):
    #    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 
    # load model
    if isinstance(model, str):
        if model == "esm2":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model == "esm1v":
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        else:
            raise ValueError(f"{model} is not a valid model")
    elif isinstance(model, torch.nn.Module):
        alphabet = torch.load(os.path.join(Path(__file__).parent, "alphabet.pt"))
    else:
        raise TypeError("Model should be either a string or a torch.nn.Module object")
 
 
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
 
    if names == None:
        names = names = [f'seq{i}' for i in range(len(seqs))]
 
    data = list(zip(names, seqs))
 
    # check datatype of sequences - str or biotite
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass  # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]
 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
 
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)
 
    return results, batch_lens, batch_labels, alphabet