from pathlib import Path
import numpy as np
import numpy.typing as npt

TRAIN_DATA = Path("/data/brunborg/tokenized/tinystory-train.npy")
VAL_DATA = Path("/data/brunborg/tokenized/tinystory-valid.npy")

VOCAB_SIZE = 10_000

embedding_dim = 512
context_length = 128

np.random.seed(42)
embeddings = np.random.randn(VOCAB_SIZE, embedding_dim)

train_dataset = np.load(TRAIN_DATA)
val_dataset = np.load(VAL_DATA)

def get_batch(
    dataset: npt.NDArray,
    context_length: int,
):
    starting_idxs = np.random.randint(0, VOCAB_SIZE, (context_length,))
    x = np.stack([
            dataset[i : i + context_length]
            for i in starting_idxs
    ])
    y = np.stack(
        [
            dataset[i + 1 : i + 1 + context_length]
            for i in starting_idxs
        ]
    )
    return embeddings[starting_idxs,:], starting_idxs

def get_loader(dataset, context_length=context_length):
    while True:
        yield get_batch(dataset, context_length)


train_loader = get_loader(train_dataset)
valid_loader = get_loader(val_dataset)

print(next(train_loader))
print(next(train_loader))