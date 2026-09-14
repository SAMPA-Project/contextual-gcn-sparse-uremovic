"""
Chronological train / validation / test split shared by every script.

Snapshots are cut in time order (no shuffling) in a fixed 4:1:1 ratio:
 - train: fit the model weights
 - val:   early stopping and best-checkpoint selection during training
 - test:  never touched by training; the only data errors are reported on
"""

TRAIN_PART, VAL_PART, TEST_PART = 4, 1, 1


def split_bounds(n):
    """
    End index of the train part and of the val part for `n` chronological
    samples: train = [0, train_end), val = [train_end, val_end),
    test = [val_end, n).
    """
    train_end = n * TRAIN_PART // (TRAIN_PART + VAL_PART + TEST_PART)
    val_end = train_end + (n - train_end) * VAL_PART // (VAL_PART + TEST_PART)
    return train_end, val_end


def train_val_test_split(dataset):
    """Split a `StaticGraphTemporalSignal` into (train, val, test) signals."""
    train_end, val_end = split_bounds(dataset.snapshot_count)
    return dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]
