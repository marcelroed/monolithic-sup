from pathlib import Path
import numpy as np
import numpy.typing as npt

TRAIN_DATA = Path("/data/brunborg/tokenized/tinystory-train.npy")
VAL_DATA = Path("/data/brunborg/tokenized/tinystory-valid.npy")

VOCAB_SIZE = 10_000

EMBEDDING_DIM = 512
CONTEXT_LENGTH = 128
SEED = 42

np.random.seed(SEED)

embeddings = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM)

train_dataset = np.load(TRAIN_DATA)
val_dataset = np.load(VAL_DATA)

def get_batch(
    dataset: npt.NDArray,
    CONTEXT_LENGTH: int,
):
    starting_idxs = np.random.randint(0, VOCAB_SIZE, (CONTEXT_LENGTH,))
    x = np.stack([
            dataset[i : i + CONTEXT_LENGTH]
            for i in starting_idxs
    ])
    y = np.stack(
        [
            dataset[i + 1 : i + 1 + CONTEXT_LENGTH]
            for i in starting_idxs
        ]
    )
    return embeddings[starting_idxs,:], starting_idxs

def get_loader(dataset, context_length=CONTEXT_LENGTH):
    while True:
        yield get_batch(dataset, context_length)


train_loader = get_loader(train_dataset)
valid_loader = get_loader(val_dataset)

print(next(train_loader))
print(next(train_loader))