import torch.distributed as dist
import random

import torch

from typing import Optional, Set, Union

from .communicate_optimize_strategy import (
    CommunicateOptimizeStrategy,
    CommunicationModule,
)
from .optim import OptimSpec
from .communicate import all_reduce, all_gather


class AveragingCommunicator(CommunicationModule):
    """
    Communication module that averages model parameters across nodes.
    Used by FedAvg strategies.
    """

    def __init__(self, island_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.island_size = island_size

    def _select_partners(self, rank: int, num_nodes: int) -> Set[int]:
        """Select partners for grouped federated averaging."""
        world_size = num_nodes

        # Only rank 0 creates the island assignments
        if rank == 0:
            ranks = list(range(world_size))
            random.shuffle(ranks)
        else:
            ranks = [None] * world_size

        dist.broadcast_object_list(ranks, src=0)

        islands = []
        island_size = self.island_size if self.island_size is not None else num_nodes
        for i in range(0, len(ranks), island_size):
            islands.append(set(ranks[i : i + island_size]))

        # Find which island this rank belongs to
        my_island = None
        for island in islands:
            if rank in island:
                my_island = island
                break

        return my_island

    def _average_models(self, model, island_members: Set[int], num_nodes: int) -> None:
        """Average model parameters across island members."""
        for param in model.parameters():
            if len(island_members) == num_nodes:
                # Full averaging - more efficient
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= num_nodes
            else:
                # Partial averaging using all_gather
                tensor_list = [torch.zeros_like(param.data) for _ in range(num_nodes)]
                all_gather(tensor_list, param.data)

                # Compute average only from ranks in the same island
                island_tensors = [tensor_list[rank] for rank in island_members]
                island_average = sum(island_tensors) / len(island_tensors)

                param.data = island_average

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform averaging communication."""
        if num_nodes > 1:
            if self.island_size is not None and self.island_size < num_nodes:
                island_members = self._select_partners(rank, num_nodes)
            else:
                island_members = set(range(num_nodes))

            self._average_models(model, island_members, num_nodes)

    def _init_node(self, model, rank, num_nodes):
        pass


class FedAvgStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        inner_optim: Optional[Union[str, OptimSpec]] = None,
        island_size: Optional[int] = None,
        H: int = 1,
        max_norm: float = None,
        **kwargs,
    ):
        # Create the averaging communicator
        averaging_comm = AveragingCommunicator(island_size=island_size)

        super().__init__(
            inner_optim=inner_optim,
            communication_modules=[averaging_comm],
            max_norm=max_norm,
            **kwargs,
        )

        self.island_size = island_size
        self.H = H

    def _communicate(self):
        """Apply communication modules at the specified frequency."""
        if self.local_step % self.H == 0 and self.local_step > 0:
            super()._communicate()

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        if self.island_size is None:
            self.island_size = num_nodes
