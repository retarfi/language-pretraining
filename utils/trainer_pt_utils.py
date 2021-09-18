from typing import Dict, List, TypeVar, Optional, Iterator
import math

import numpy as np
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from transformers.file_utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled, is_torch_tpu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import IterableDatasetShard


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)

def get_indices_per_process(
        indices:List, per_device_batch_size:int, 
        batch_config:Dict, real_batch_size:int, 
        node_rank:int, num_batch:int, 
        rank:int, nproc_per_node:int
    ) -> List:
    num_before = sum([batch_config[str(i)] for i in range(node_rank)]) + per_device_batch_size * (rank - node_rank * nproc_per_node)
    num_after = num_before + per_device_batch_size
    base_idx = np.arange(num_before, num_after)
    expand_idx = np.array([base_idx + i*real_batch_size for i in range(num_batch)]).reshape(per_device_batch_size*num_batch)
    indices = np.array(indices)[expand_idx].tolist()
    return indices


# Override PyTorch class DistributedSampler
class MyDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, per_device_batch_size: int, 
                 batch_config: dict, node_rank: int,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.per_device_batch_size = per_device_batch_size
        self.batch_config = batch_config
        self.real_batch_size = sum(self.batch_config.values())
        self.node_rank = node_rank
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.nproc_per_node = int(batch_config[str(self.node_rank)] / self.per_device_batch_size)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.real_batch_size != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_batch_samples = math.ceil((len(self.dataset) - self.real_batch_size) / self.real_batch_size)
        else:
            self.num_batch_samples = math.ceil(len(self.dataset) / self.real_batch_size)
        self.num_samples = self.num_batch_samples * self.per_device_batch_size
        self.total_size = self.num_batch_samples * self.real_batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        '''
        CONSIDER UNBALANCE BATCH
        '''
        indices = get_indices_per_process(
            indices=indices, per_device_batch_size=self.per_device_batch_size,
            batch_config=self.batch_config, real_batch_size=self.real_batch_size,
            node_rank=self.node_rank, num_batch=self.num_batch_samples,
            rank=self.rank, nproc_per_node=self.nproc_per_node
        )
        assert len(indices) == self.num_samples, f'{len(indices)} != {self.num_samples}'

        return iter(indices)
