import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

BATCH_SIZE = 2
VOCAB_SIZE = 256
EMBED_DIM = 16
CTX_LEN = 10

embedding_table = torch.randn(VOCAB_SIZE, EMBED_DIM)

#  def embedding_forward(input_ids, embedding_table):
#      embeddings = torch.gather(
#          embedding_table.unsqueeze(0).expand(input_ids.shape[0], -1, -1),
#          dim=1,
#          index=input_ids.unsqueeze(-1).expand(-1, -1, embedding_table.shape[1])
#      )
#      return embeddings

#  def embedding_forward(input_ids, embedding_table):
#      batch_size, seq_len = input_ids.shape
#      vocab_size, embed_dim = embedding_table.shape
#      token_indices = torch.repeat_interleave(input_ids, embed_dim)
#      dim_indices = torch.arange(embed_dim, device=input_ids.device).repeat(batch_size*seq_len)
#      flat_embeddings = torch.gather(
#          embedding_table.flatten(),
#          dim=0,
#          index=dim_indices + token_indices * embed_dim
#      )
#      embeddings = flat_embeddings.view(batch_size, seq_len, embed_dim)
#      return embeddings

def embedding_forward(input_ids, embedding_table):
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = embedding_table.shape
    token_indices = torch.repeat_interleave(input_ids.view(-1, 1), embed_dim, dim=1)
    flat_embeddings = torch.gather(
        embedding_table,
        dim=0,
        index=token_indices,
    )
    embeddings = flat_embeddings.view(batch_size, seq_len, embed_dim)
    return embeddings

def embedding_backward(grad_output, input_ids, vocab_size):
    _, _, embed_dim = grad_output.shape
    grad_embedding = torch.zeros(vocab_size, embed_dim, device=grad_output.device)
    grad_embedding.scatter_add_(
        dim=0,
        index=input_ids.view(-1).unsqueeze(-1).expand(-1, embed_dim),
        src=grad_output.view(-1, embed_dim)
    )
    return grad_embedding

input_ids = torch.randint(0, 10, (BATCH_SIZE, CTX_LEN))
embeddings = embedding_forward(input_ids, embedding_table)
input_ids.shape
embedding_table[input_ids].shape
assert (embeddings == embedding_table[input_ids]).all()
# +
grad_embedding = embedding_backward(embeddings, input_ids, embedding_table.shape[0])

grad_embedding

# reproducibility
torch.manual_seed(0)

# your original table and inputs
vocab_size, embed_dim = 256, 16
embedding_table = torch.randn(vocab_size, embed_dim)
input_ids = torch.tensor([
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    [0, 10,  2,  3,  4,  5,  6,  7,  8,  9],
])

# ----------------------------------------------------------------------------
# 1) Using torch.nn.functional.embedding
# ----------------------------------------------------------------------------
# make a fresh copy of the table with requires_grad
emb_tbl1 = embedding_table.clone().requires_grad_(True)

# forward
embeddings1 = F.embedding(input_ids, emb_tbl1)

# backward: use the same "grad_output" you used before (here we used embeddings itself)
embeddings1.backward(embeddings1)

# grab the gradient w.r.t. the table
grad1 = emb_tbl1.grad


# ----------------------------------------------------------------------------
# 2) Using torch.nn.Embedding
# ----------------------------------------------------------------------------
# create an Embedding module initialized to your table
emb_mod = nn.Embedding(vocab_size, embed_dim)
with torch.no_grad():
    emb_mod.weight.copy_(embedding_table)      # set weights
emb_mod.weight.requires_grad_(True)

# forward
embeddings2 = emb_mod(input_ids)

# backward with same grad_output
embeddings2.backward(embeddings2)

# gradient from the module
grad2 = emb_mod.weight.grad


# ----------------------------------------------------------------------------
# 3) Compare against your manual scatter_add implementation
# ----------------------------------------------------------------------------
# suppose grad_manual is what you computed with your `embedding_backward`
# from your snippet:
#    grad_manual = embedding_backward(embeddings, input_ids, vocab_size)
# (we’ll recompute it here for clarity)

#  def embedding_backward(grad_output, input_ids, vocab_size):
#      batch, seq_len, D = grad_output.shape
#      grad_emb = torch.zeros(vocab_size, D, device=grad_output.device)
#      grad_emb.scatter_add_(
#          dim=0,
#          index=input_ids.view(-1).unsqueeze(-1).expand(-1, D),
#          src=grad_output.view(-1, D),
#      )
#      return grad_emb

grad_manual = embedding_backward(embeddings1, input_ids, vocab_size)


# Now check that all three gradients are identical:
print("F.embedding vs manual:", torch.allclose(grad1, grad_manual))
print("nn.Embedding vs manual:", torch.allclose(grad2, grad_manual))
print("F.embedding vs nn.Embedding:",  torch.allclose(grad1, grad2))
