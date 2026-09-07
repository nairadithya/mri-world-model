"""Length-bucketed batch sampler.

Motivation: collate pads every batch to its longest patient, materializing
dense (B, T_max, M, 1, 96, 96, 96) float32 (~284 MB at B=1/T=21, ~1.1 GB at
B=4/T=21). Under uniform shuffle, long+short pairings maximize padding waste
and assemble the OOM worst case (a long-history patient sharing a batch).
Bucketing keeps T_max small per batch and stabilizes per-step pair counts,
which also conditions the pair-weighted accumulation windows in trainer.py.

Deterministic given (seed, iteration count): __iter__ draws from
random.Random(seed + n_calls). num_workers>0 is safe (the sampler is consumed
in the main process). Use for TRAIN only; keep val/test sequential.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from torch.utils.data import Sampler


class LengthBucketSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.lengths = list(lengths)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._calls = 0

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        idx.sort(key=lambda i: self.lengths[i])
        batches = [
            idx[i:i + self.batch_size]
            for i in range(0, len(idx), self.batch_size)
        ]
        if self.drop_last and len(batches) > 1 and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            rng = random.Random(self.seed + self._calls)
            rng.shuffle(batches)
        self._calls += 1
        return iter(batches)

    def __len__(self) -> int:
        full, rem = divmod(len(self.lengths), self.batch_size)
        n = full + (1 if rem else 0)
        if self.drop_last and rem and n > 1:
            n -= 1
        return n
