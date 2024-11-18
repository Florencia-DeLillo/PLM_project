import torch
import esm
import random

def mask_seq(batch, batch_size, n):
    """
    Performs masking on sequence
        batch: (item of DataLoader) 
        n: (int) batch iteration for random.seed() control
    Masking rules of 15% of each sequence:
        80%: mask
        10%: random mutation
        10%: unchanged
    """
    
    masked_batch = []
    vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']

    for i, item in enumerate(batch):
        seq = item['sequence']
        random.seed(n * batch_size + i)
        chain = list(seq)
        seq_len = len(chain)
        num_to_mask = max(1, int(round(0.15 * seq_len)))
        mask_indices = random.sample(range(seq_len), num_to_mask)
        
        num_mask = int(round(0.8 * num_to_mask))
        num_random = int(round(0.1 * num_to_mask))
        num_unchanged = num_to_mask - num_mask - num_random

        # Adjust counts if necessary to ensure they sum to num_to_mask
        while num_mask + num_random + num_unchanged > num_to_mask:
            num_mask -= 1
        while num_mask + num_random + num_unchanged < num_to_mask:
            num_mask += 1

        # Create action labels
        actions = ['mask'] * num_mask + ['random'] * num_random + ['unchanged'] * num_unchanged
        random.shuffle(actions)

        # Apply masking according to actions
        for idx, action in zip(mask_indices, actions):
            if action == 'mask':
                chain[idx] = '<mask>'
            elif action == 'random':
                chain[idx] = random.choice(vocab)
            else:
                pass  # Leave the token unchanged

        masked_batch.append(' '.join(chain))
    
    return masked_batch


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
    

