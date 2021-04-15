import itertools
from typing import Optional

import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from .comm import get_world_size, get_rank, shared_random_seed


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = get_rank()
        self._world_size = get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset, sampler):
        """
        Args:
            dataset (torch.utils.data.Dataset): an old-style dataset with ``__getitem__``
            sampler (torch.utils.data.sampler.Sampler): a cheap iterable that produces indices
                to be applied on ``dataset``.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            for idx in self.sampler:
                yield self.dataset[idx]
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker and only keep 1/N of the ids on each
            # worker. The assumption is that sampler is cheap to iterate and it's fine to discard
            # ids in workers.
            for idx in itertools.islice(
                    self.sampler, worker_info.id, None, worker_info.num_workers
            ):
                yield self.dataset[idx]
