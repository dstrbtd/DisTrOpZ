import math
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist

from copy import deepcopy
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Type, Union, Optional, Dict, Any
from abc import ABC, abstractmethod

from exogym.aux.utils import LogModule


# COMMUNICATION
def mps_compatible(func):
    # Wrapper for all_gather which handles tensor_list and tensor
    def all_gather_wrapper(tensor_list, tensor, *args, **kwargs):
        # Check if either is on MPS
        is_tensor_mps = hasattr(tensor, "device") and tensor.device.type == "mps"
        is_list_mps = any(
            hasattr(t, "device") and t.device.type == "mps" for t in tensor_list
        )

        if is_tensor_mps or is_list_mps:
            # Convert tensor to CPU if needed
            if is_tensor_mps:
                cpu_tensor = tensor.data.to("cpu")
            else:
                cpu_tensor = tensor

            # Convert tensor_list to CPU if needed
            cpu_tensor_list = []
            for t in tensor_list:
                if hasattr(t, "device") and t.device.type == "mps":
                    cpu_tensor_list.append(t.data.to("cpu"))
                else:
                    cpu_tensor_list.append(t)

            # Call function with CPU tensors
            result = func(cpu_tensor_list, cpu_tensor, *args, **kwargs)

            # Copy data back to original devices
            if is_tensor_mps:
                tensor.data.copy_(cpu_tensor.to("mps"))

            for i, t in enumerate(tensor_list):
                if hasattr(t, "device") and t.device.type == "mps":
                    t.data.copy_(cpu_tensor_list[i].to("mps"))

            return result
        else:
            return func(tensor_list, tensor, *args, **kwargs)

    # Wrapper for all other functions that handle a single tensor
    def standard_wrapper(tensor, *args, **kwargs):
        if hasattr(tensor, "device") and tensor.device.type == "mps":
            # Move the tensor to CPU
            cpu_tensor = tensor.data.to("cpu")
            # Call the function on CPU
            result = func(cpu_tensor, *args, **kwargs)
            # Copy the result back to mps
            tensor.data.copy_(cpu_tensor.to("mps"))
            return result
        else:
            return func(tensor, *args, **kwargs)

    # Return the appropriate wrapper based on function name
    if func.__name__ == "all_gather":
        return all_gather_wrapper
    else:
        return standard_wrapper


@mps_compatible
def broadcast(tensor, src=0):
    return dist.broadcast(tensor, src=src)


@mps_compatible
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    return dist.all_reduce(tensor, op=op)


@mps_compatible
def all_gather(tensor_list, tensor, group=None, async_op=False):
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


# @mps_compatible
# def reduce_scatter(tensor):
#     return dist.reduce_scatter(tensor)

# @mps_compatible
# def reduce(tensor):
#     return dist.reduce(tensor)

# @mps_compatible
# def gather(tensor):
#     return dist.gather(tensor)


# OPTIMIZER
@dataclass
class OptimSpec:
    cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    kwargs: Dict[str, Any] = None  # e.g. {'lr': 3e-4}

    def __init__(self, cls: Type[torch.optim.Optimizer], **kwargs: Dict[str, Any]):
        self.cls = cls
        self.kwargs = kwargs

    @classmethod
    def from_string(cls, name: str, **kwargs) -> "OptimSpec":
        """Create OptimSpec from optimizer name string."""
        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        name_lower = name.lower()
        if name_lower not in optimizer_map:
            available = ", ".join(optimizer_map.keys())
            raise ValueError(
                f"Unknown optimizer '{name}'. Available options: {available}"
            )

        return cls(optimizer_map[name_lower], **kwargs)

    def build(self, model):
        return self.cls(model.parameters(), **(self.kwargs or {}))


def ensure_optim_spec(
    optim: Union[str, OptimSpec, None], default: Optional[OptimSpec] = None, **kwargs
) -> OptimSpec:
    """Convert string or OptimSpec to OptimSpec instance."""
    if optim is None:
        if default is None:
            return OptimSpec(torch.optim.AdamW, **kwargs)
        else:
            return default
    elif isinstance(optim, str):
        return OptimSpec.from_string(optim, **kwargs)
    elif isinstance(optim, OptimSpec):
        # If additional kwargs provided, merge them
        if kwargs:
            merged_kwargs = {**(optim.kwargs or {}), **kwargs}
            return OptimSpec(optim.cls, **merged_kwargs)
        return optim
    else:
        raise TypeError(f"Expected str, OptimSpec, or None, got {type(optim)}")


# STRATEGY
class Strategy(ABC, LogModule):
    def __init__(
        self,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
        **kwargs: Dict[str, Any],
    ):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.kwargs = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize scheduler as None; will be set after self.optim is defined in subclasses.
        self.scheduler = None

        # List of callbacks to record learning rate changes.
        self.lr_callbacks = []

        self.max_steps = 1  # Needs to be initialized for first call of lr_lambda.

    def _init_node(self, model, rank, num_nodes):
        self.model = model
        self.rank = rank
        self.num_nodes = num_nodes

        self.local_step = 0

        if hasattr(self, "optim_spec"):
            self.optim = self.optim_spec.build(model)

    @abstractmethod
    def step(self):
        self.nbytes = 0

        if self.scheduler is not None:
            self.scheduler.step()

            if self.rank == 0:
                for callback in self.lr_callbacks:
                    callback(self.scheduler.get_last_lr()[0])

        self.local_step += 1

    def zero_grad(self):
        self.optim.zero_grad()

    def _setup_scheduler(self):
        def lr_lambda(current_step):
            warmup_steps = self.lr_scheduler_kwargs.get("warmup_steps", 1)
            # If max steps not set,
            if "max_steps" in self.lr_scheduler_kwargs:
                max_steps = min(self.lr_scheduler_kwargs["max_steps"], self.max_steps)
            else:
                max_steps = self.max_steps
            cosine_anneal = self.lr_scheduler_kwargs.get("cosine_anneal", False)

            if current_step < warmup_steps:
                return float(current_step) / float(max(warmup_steps, 1))
            elif cosine_anneal:
                min_lr_factor = 0.1
                progress = (current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                return 1.0

        if self.lr_scheduler == "lambda_cosine":
            self.scheduler = LambdaLR(self.optim, lr_lambda)
        elif self.lr_scheduler is not None:
            lr_sched_kwargs = (
                self.lr_scheduler_kwargs if self.lr_scheduler_kwargs is not None else {}
            )
            self.scheduler = self.lr_scheduler(self.optim, **lr_sched_kwargs)
        else:
            self.scheduler = None

    def __config__(self):
        remove_keys = [
            "iteration",
            "local_step",
            "lr_callbacks",
            "model",
            "optim",
            "scheduler",
        ]

        config = super().__config__(remove_keys)

        config["strategy"] = self.__class__.__name__

        return config


class SimpleReduceStrategy(Strategy):
    def __init__(self, optim_spec=None, max_norm=None, **kwargs):
        super().__init__(**kwargs)

        self.optim_spec = ensure_optim_spec(optim_spec) or OptimSpec(torch.optim.AdamW)

        self.max_norm = max_norm

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()

    def step(self):
        if self.num_nodes > 1 or True:
            for param in self.model.parameters():
                if param.grad is not None:
                    all_reduce(param.grad)
                    param.grad.div_(self.num_nodes)

            if self.max_norm:
                nn_utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )

        self.optim.step()

        super().step()


# COMMUNICATE OPTIMIZER STRATEGY
class CommunicationModule(ABC):
    """Abstract base class for communication modules."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """
        Perform communication for the given model.

        Args:
          model: The model to communicate
          rank: Current node rank
          num_nodes: Total number of nodes
          local_step: Current local step count
        """
        pass

    @abstractmethod
    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """
        Initialize the communication module for the given model.
        """
        pass


class CommunicateOptimizeStrategy(Strategy):
    """
    Base class for strategies that interleave communication and optimization.

    This strategy:
    1. Performs local optimization step
    2. Applies communication modules when the derived strategy decides
    """

    def __init__(
        self,
        communication_modules: List[CommunicationModule],
        optim_spec: Optional[Union[str, OptimSpec]] = None,
        max_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.optim_spec = ensure_optim_spec(optim_spec) or OptimSpec(torch.optim.AdamW)

        self.communication_modules = communication_modules
        self.max_norm = max_norm

        # Set strategy reference in communication modules that need it
        for comm_module in self.communication_modules:
            comm_module.strategy = self

    def step(self):
        # Gradient clipping if specified
        if self.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

        # Local optimization step
        self.optim.step()

        # Communication phase - let derived strategies decide when
        self._communicate()

        super().step()

    def _communicate(self):
        """Apply all communication modules sequentially. Override in derived classes for custom timing."""
        for comm_module in self.communication_modules:
            comm_module.communicate(
                self.model, self.rank, self.num_nodes, self.local_step
            )

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        for comm_module in self.communication_modules:
            comm_module._init_node(model, rank, num_nodes)

        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()


class SparseCommunicator(CommunicationModule):
    """
    Communication module for sparse parameter communication (like SPARTA).
    """

    def __init__(self, index_selector, **kwargs):
        super().__init__(**kwargs)
        self.index_selector = index_selector
        self.iteration = 0

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform sparse communication."""
        if num_nodes > 1:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue

                    indices_mask = self.index_selector.get_indices(
                        param, self.iteration
                    )

                    # Broadcasting a mask might be needed
                    broadcast(indices_mask, src=0)
                    sparse_data = param.data[indices_mask]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= num_nodes

                    param.masked_scatter_(indices_mask, sparse_data)

        self.iteration += 1

    def _init_node(self, model, rank, num_nodes):
        pass


class SPARTAStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        optim_spec: Optional[Union[str, OptimSpec]] = None,
        p_sparta=0.005,
        **kwargs,
    ):
        # Create index selector and sparse communicator
        index_selector = RandomIndexSelector(p_sparta)
        sparse_comm = SparseCommunicator(index_selector)

        super().__init__(
            optim_spec=optim_spec, communication_modules=[sparse_comm], **kwargs
        )

        self.index_selector = index_selector


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    # Add iteration argument to the base class signature
    def get_indices(self, param, iteration):
        # Default implementation returns all indices (mask of Trues)
        return torch.ones_like(param, dtype=torch.bool)


class RandomIndexSelector(IndexSelector):
    # Update signature to match base class
    def get_indices(self, param, iteration):
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()


class ShuffledSequentialIndexSelector(IndexSelector):
    def __init__(self, p):
        # No model-dependent init here
        super().__init__(p)
        # Remove self.shuffled_state and self.index

    # Update signature to match base class
    def get_indices(self, param, iteration):
        num_total = param.numel()
        if num_total == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        # Initialize state for this parameter if not seen before
        if param not in self.state:
            num_partitions = max(
                1, math.ceil(1.0 / self.p)
            )  # Ensure at least 1 partition
            shuffled_indices = torch.randperm(num_total, device=param.device)
            self.state[param] = {
                "num_partitions": num_partitions,
                "shuffled_indices": shuffled_indices,
            }

        param_state = self.state[param]
        num_partitions = param_state["num_partitions"]
        shuffled_indices = param_state["shuffled_indices"]

        # Determine the current chunk based on the iteration number
        current_chunk = iteration % num_partitions

        # Calculate chunk size and remainder for potentially uneven distribution
        chunk_size = num_total // num_partitions
        remainder = num_total % num_partitions

        # Calculate start and end indices for the current chunk
        start_index = current_chunk * chunk_size + min(current_chunk, remainder)
        # The end index calculation ensures the chunk size is correct, adding 1 for chunks getting the remainder
        end_index = start_index + chunk_size + (1 if current_chunk < remainder else 0)

        # Get the flat indices for the current chunk
        selected_flat_indices = shuffled_indices[start_index:end_index]

        # Create and return the boolean mask
        mask = torch.zeros(num_total, dtype=torch.bool, device=param.device)
        if (
            selected_flat_indices.numel() > 0
        ):  # Handle empty selection if num_total is very small
            mask[selected_flat_indices] = True
        return mask.view(param.shape)


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)
        # Note: This class implicitly uses a step counter per parameter via self.state[param]["curr_partition"]
        # It doesn't need the global iteration number passed in.
        # To be consistent, we should update its signature, but the iteration argument would be unused.

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        # Ensure at least 1 partition
        num_partitions = max(1, min(math.ceil(1.0 / self.p), param.numel()))
        param_state["num_partitions"] = num_partitions
        if param.numel() > 0:
            param_state["partitions"] = (
                torch.rand(param.numel(), device=param.device).argsort()
                % num_partitions
            )
        else:
            # Handle zero-element tensors
            param_state["partitions"] = torch.empty(
                0, dtype=torch.long, device=param.device
            )

    # Update signature, though iteration is unused here
    def get_indices(self, param, iteration):
        # Handle zero-element tensors gracefully
        if param.numel() == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        # Check if cycle needs reset BEFORE accessing partitions
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        param_state = self.state[param]

        # Need to handle case where num_partitions might be 0 if numel was 0 during _set_partition
        # Although we added checks for numel=0, ensure partition access is safe
        if param_state["num_partitions"] == 0:
            return torch.zeros_like(
                param, dtype=torch.bool
            )  # Should not happen if numel > 0

        # Indices calculation requires reshaping the flat partitions result
        partition_indices = param_state["partitions"] == param_state["curr_partition"]
        indices_mask = partition_indices.view(
            param.shape
        ).bool()  # Reshape flat bool tensor to param shape

        param_state["curr_partition"] += 1

        return indices_mask


class SPARTADiLoCoStrategy(CommunicateOptimizeStrategy):
    """
    Strategy that combines SPARTA's sparse communication with DiLoCo's master-worker optimization.

    This strategy:
    1. Performs local optimization
    2. Applies sparse communication every sparta_interval steps (SPARTA)
    3. Applies master-worker optimization every H steps (DiLoCo)
    """

    def __init__(
        self,
        inner_optim_spec: Optional[Union[str, OptimSpec]] = None,
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        p_sparta: float = 0.005,
        sparta_interval: int = 1,
        H: int = 100,
        **kwargs,
    ):
        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(inner_optim_spec, OptimSpec(torch.optim.AdamW))

        # Create both communication modules
        index_selector = RandomIndexSelector(p_sparta)
        self.sparse_comm = SparseCommunicator(index_selector)
        self.diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)

        # Store timing parameters
        self.sparta_interval = sparta_interval
        self.H = H

        super().__init__(
            optim_spec=optim_spec,
            communication_modules=[self.sparse_comm, self.diloco_comm],
            **kwargs,
        )

        self.index_selector = index_selector


class DiLoCoCommunicator(CommunicationModule):
    """
    Communication module for master-worker setup (like DiLoCo).
    """

    def __init__(
        self,
        H: int = 100,
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        **kwargs,
    ):
        self.H = H
        self.outer_optim_spec = ensure_optim_spec(
            outer_optim_spec,
            OptimSpec(torch.optim.SGD, lr=0.7, nesterov=True, momentum=0.9),
        )
        self.strategy = None  # Will be set by CommunicateOptimizeStrategy
        self.master_model = None
        self.outer_optimizer = None

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform master-worker communication."""
        if num_nodes > 1 and local_step % self.H == 0 and local_step > 0:
            # First average all models
            for param in model.parameters():
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= num_nodes

            # Master does outer optimization step
            if rank == 0 and self.master_model is not None:
                self.outer_optimizer.zero_grad()
                self._set_master_grad(model)
                self.outer_optimizer.step()
                self._synchronize_master_model(model)

            # Broadcast updated parameters
            for param in model.parameters():
                broadcast(param.data, src=0)

    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """Initialize master model for rank 0."""
        if rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True
            self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

    def _set_master_grad(self, model) -> None:
        """Set gradients on master model based on difference between master and worker models."""
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - model.state_dict()[name].data.to("cpu")

    def _synchronize_master_model(self, model) -> None:
        """Synchronize worker model with master model parameters."""
        for name, param in model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)


STRATEGY = SPARTAStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 1000,
        "cosine_anneal": True,
    },
    max_norm=1.0,
    H=5,
)
