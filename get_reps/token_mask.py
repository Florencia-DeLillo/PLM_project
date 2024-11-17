import torch
import esm

def mask_tokens(tokens: torch.Tensor, alphabet: esm.Alphabet, n: int):
    """
    Performs masking on tokens
        tokens: (torch.Tensor) output of alphabet.get_batch_converter()(data)
        alphabet (esm.Alphabet) vocabulary used for tokenization
        n (int): batch iteration for random.seed() control
    Masking rules of 15% of each sequence:
        80%: mask
        10%: random mutation
        10%: unchanged
    """

    assert isinstance(tokens, torch.Tensor) == True
    assert isinstance(alphabet, esm.Alphabet) == True

    raise NotImplementedError

    vocab = alphabet.all_toks
    special = alphabet.all_special_tokens
    TOK2VOCAB = {tok: n for n, tok in enumerate(vocab)}

    # Get special token indices
    special_toks = [TOK2VOCAB[tok] for tok in special]
    sequence_toks = [TOK2VOCAB[tok] for tok in vocab if tok not in special_toks]

    # Create a filter for maskable
    maskable_positions = torch.ones_like(tokens, dtype=torch.bool)
    for special_id in special_toks:
        maskable_positions &= tokens != special_id

    batch_size, seq_len = tokens.size()
    masked_indices = torch.zeros_like(tokens, dtype=torch.bool)

    

    

