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

# override transformers.trainer_pt_utils.get_length_grouped_indices
def get_length_grouped_indices(
        lengths:List,
        batch_config:Dict,
        generator:Optional[torch.Generator] = None
    ) -> List:
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)

    if set(batch_config.values()) == 1:
        batch_size = list(batch_config.values())[0]
        # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1
        megabatch_size = mega_batch_mult * batch_size
    else:
        megabatch_size = np.lcm.reduce(list(batch_config.values()))
        if megabatch_size < 2000:
            megabatch_size = 2000 // megabatch_size * megabatch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


def get_indices_per_process(
        indices:List, batch_size:int, 
        batch_config:Dict, real_batch_size:int, 
        node_rank:int, num_batch:int, 
        rank:int, nproc_per_node:int
    ) -> List:
    num_before = sum([batch_config[str(i)] for i in range(node_rank)]) + batch_size * (rank - node_rank * nproc_per_node)
    num_after = num_before + batch_size
    base_idx = np.arange(num_before, num_after)
    expand_idx = np.array([base_idx + i for i in range(num_batch)]).reshape(batch_size*num_batch)
    indices = np.array(indices)[expand_idx].tolist()
    return indices


# Override
class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        real_batch_size:int,
        batch_config: dict,
        node_rank: int,
        num_replicas: Optional[int] = None,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.real_batch_size = real_batch_size
        self.batch_config = batch_config
        self.num_replicas = num_replicas
        if len(self.batch_config.keys()) != self.num_replicas:
            raise ValueError(f'There should be {len(self.batch_config.keys())} batch-size in parameter file, but got {self.num_replicas}')
        self.node_rank = node_rank
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.nproc_per_node = int(batch_config[str(self.node_rank)] / self.batch_size)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.real_batch_size != 0:
        #if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_batch_samples = math.ceil((len(self.dataset) - self.real_batch_size) / self.real_batch_size)
            #self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_batch_samples = math.ceil(len(self.dataset) / self.real_batch_size)
        self.num_samples = self.num_batch_samples * self.batch_size
        self.total_size = self.num_batch_samples * self.real_batch_size
        #self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.model_input_name = model_input_name if model_input_name is not None else "input_ids"

        if lengths is None:
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or self.model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{self.model_input_name}' key."
                )
            lengths = [len(feature[self.model_input_name]) for feature in dataset]
        self.lengths = lengths

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_config, generator=g)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        '''
        CONSIDER UNBALANCE BATCH
        '''
        # indices = indices[self.rank : self.total_size : self.num_replicas]
        indices = get_indices_per_process(
            indices=indices, batch_size=self.batch_size,
            batch_config=self.batch_config, real_batch_size=self.real_batch_size,
            node_rank=self.node_rank, num_batch=self.num_batch_samples,
            rank=self.rank, nproc_per_node=self.nproc_per_node
        )
        assert len(indices) == self.num_samples

        return iter(indices)


# Override PyTorch class DistributedSampler
class MyDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size: int, real_batch_size:int, 
                 batch_config: dict, node_rank: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.real_batch_size = real_batch_size
        self.batch_size = batch_size
        self.batch_config = batch_config
        self.num_replicas = num_replicas
        self.node_rank = node_rank
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.nproc_per_node = int(batch_config[str(self.node_rank)] / self.batch_size)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.real_batch_size != 0:
        #if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_batch_samples = math.ceil((len(self.dataset) - self.real_batch_size) / self.real_batch_size)
            #self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
            #    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            #)
        else:
            self.num_batch_samples = math.ceil(len(self.dataset) / self.real_batch_size)
            #self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.num_samples = self.num_batch_samples * self.batch_size
        self.total_size = self.num_batch_samples * self.real_batch_size
        #self.total_size = self.num_samples * self.num_replicas
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
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = get_indices_per_process(
            indices=indices, batch_size=self.batch_size,
            batch_config=self.batch_config, real_batch_size=self.real_batch_size,
            node_rank=self.node_rank, num_batch=self.num_batch_samples,
            rank=self.rank, nproc_per_node=self.nproc_per_node
        )
        assert len(indices) == self.num_samples, f'{len(indices)} != {self.num_samples}'

        return iter(indices)


# Override IterableDatasetShard
class MyIterableDatasetShard(IterableDatasetShard):
    def __init__(
        self,
        dataset: IterableDataset,
        real_batch_size: int,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            drop_last = batch_size,
            num_processes = num_processes,
            process_index = process_index,
            seed = seed
        )
        self.real_batch_size = real_batch_size

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        # real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == self.real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < self.real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]
