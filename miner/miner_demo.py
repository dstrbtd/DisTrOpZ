import math
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist
import torch.fft

from copy import deepcopy
from dataclasses import dataclass
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Type, Union, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from exogym.aux.utils import LogModule


# BASE COMMUNICATION
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


# BASE OPTIMIZER SPEC
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


# BASE STRATEGY
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


# BASE COMMUNICATE OPTIMIZER STRATEGY
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


# OPTIMIZER

# DeMo: Decoupled Momentum Optimization
#
# Paper:
# Bowen Peng, Jeffrey Quesnelle, Diederik P. Kingma.
# “DeMo: Decoupled Momentum Optimization.”
# arXiv preprint arXiv:2411.19870, 2024. https://arxiv.org/abs/2411.19870 :contentReference[oaicite:0]{index=0}
#
# Code repository:
# https://github.com/bloc97/DeMo
#
# If you use this implementation, please cite:
#
#   @article{peng2024demo,
#     title   = {DeMo: Decoupled Momentum Optimization},
#     author  = {Peng, Bowen and Quesnelle, Jeffrey and Kingma, Diederik P.},
#     journal = {arXiv preprint arXiv:2411.19870},
#     year    = {2024},
#     url     = {https://arxiv.org/abs/2411.19870}
#   }

"""DeMo: Decoupled Momentum Optimization

This implements the DeMo fused optimizer and data parallel algorithm.
It is recommended to use DeMo as the base data parallelism.
In an exisiting codebase that uses PyTorch DDP, wrap your forward-backward in
`torch.distributed.DistributedDataParallel.no_sync` to disable external gradient synchronization.
See https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
"""


class DeMo(torch.optim.SGD):
    def __init__(
        self,
        params,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        process_group: Optional[dist.ProcessGroup] = None,
        custom_all_gather=None,
        **kwargs,
    ):
        super().__init__(
            params,
            foreach=False,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            weight_decay=0.0,
            **kwargs,
        )

        if not custom_all_gather:
            self.all_gather = dist.all_gather
        else:
            self.all_gather = custom_all_gather

        self.compression_decay = compression_decay
        self.compression_chunk = compression_chunk
        self.compression_topk = compression_topk
        self.process_group = process_group
        self.weight_decay = weight_decay

        if self.compression_topk <= 0:
            raise ValueError("topk_size has to be positive")
        if self.compression_chunk <= 0:
            raise ValueError("chunk_size has to be positive")
        if self.compression_decay < 0:
            raise ValueError("Negative compression_decay is currently not supported")
        if self.compression_decay >= 1:
            raise ValueError(
                "Values of compression_decay bigger or equal to 1.0 is currently not supported"
            )

        self.demo_state = {}
        self._init_demo_states()
        self._init_opt_parameters()

        self.default_dtype = self._find_dtype()
        self.transform = TransformDCT(self.param_groups, self.compression_chunk)
        self.compress = CompressDCT()

    def _find_dtype(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    return p.dtype
        return torch.float32

    def _init_demo_states(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.demo_state[p] = {}

    def _state_parameter(self, p):
        if p not in self.demo_state:
            self.demo_state[p] = {}
        return self.demo_state[p]

    def _init_opt_parameters(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self._state_parameter(p)

                    state["step"] = 0
                    state["delta"] = torch.zeros_like(p)

    def _demo_all_gather(self, sparse_idx, sparse_val):
        world_size = (
            dist.get_world_size()
            if self.process_group is None
            else self.process_group.size()
        )

        # Gather all the idx and vals
        sparse_idx_list = [torch.zeros_like(sparse_idx) for wi in range(world_size)]
        sparse_val_list = [torch.zeros_like(sparse_val) for wi in range(world_size)]

        sparse_idx_handle = self.all_gather(
            sparse_idx_list, sparse_idx, group=self.process_group, async_op=True
        )
        sparse_val_handle = self.all_gather(
            sparse_val_list, sparse_val, group=self.process_group, async_op=True
        )

        sparse_idx_handle.wait()
        sparse_val_handle.wait()

        return sparse_idx_list, sparse_val_list

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        self.data_transmit = 0
        self.data_receive = 0

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._state_parameter(p)

                # Update step
                state["step"] += 1

                # Step-Weight decay
                if self.weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self.weight_decay)

                # Decay delta
                if self.compression_decay != 1:
                    state["delta"].mul_(self.compression_decay)

                # Add delta to new gradient
                state["delta"].add_(p.grad, alpha=lr)

                # Compress delta
                sparse_idx, sparse_val, xshape, totalk = self.compress.compress(
                    self.transform.encode(state["delta"]), self.compression_topk
                )

                # Estimate transmitted delta
                transmit_grad = self.transform.decode(
                    self.compress.decompress(p, sparse_idx, sparse_val, xshape, totalk)
                )

                # Remove transmitted from delta
                state["delta"].sub_(transmit_grad)

                # All-gather
                sparse_idx_gather, sparse_val_gather = self._demo_all_gather(
                    sparse_idx, sparse_val
                )

                # Log I/O data size
                self.data_transmit += sparse_idx.nbytes + sparse_val.nbytes
                for si, v in zip(sparse_idx_gather, sparse_val_gather):
                    self.data_receive += si.nbytes + v.nbytes

                # Decode grad from all nodes
                new_grad = self.transform.decode(
                    self.compress.batch_decompress(
                        p, sparse_idx_gather, sparse_val_gather, xshape, totalk
                    )
                )

                # Set grad to values
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)

                # Sign-SGD
                p.grad.sign_()

        # SGD step
        return super().step(closure)


class TransformDCT:
    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    # Get the closest smallest divisor to the targeted DCT size
                    sc = _get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    # Pregenerate DCT basis matrices
                    if sc not in self.f_dict:
                        identity = torch.eye(sc)
                        self.f_dict[sc] = (
                            _dct(identity, norm=norm).to(p.dtype).to(p.device)
                        )
                        self.b_dict[sc] = (
                            _idct(identity, norm=norm).to(p.dtype).to(p.device)
                        )

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            # print(x.shape)
            # print(n1, n2)
            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:  # 2D weights
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device)
            self.b_dict[n1] = n1w

            x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class CompressDCT:
    @torch.no_grad()
    def __init__(self):
        pass

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x, topk):
        xshape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx = torch.topk(x.abs(), k=topk, dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)

        return idx, val, xshape, totalk

    @torch.no_grad()
    def decompress(self, p, idx, val, xshape, totalk):
        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # TODO: Careful, this is nondeterministic across different CUDA devices! might cause errors to accumulate between nodes!
        x.scatter_reduce_(
            dim=-1, index=idx, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])

        return x

    @torch.no_grad()
    def batch_decompress(self, p, idx, val, xshape, totalk):
        idx = torch.concatenate(idx, dim=-1).to(device=p.device)
        val = torch.concatenate(val, dim=-1).to(device=p.device)
        return self.decompress(p, idx, val, xshape, totalk)


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n):
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n):
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


## TODO: This is really slow at the moment...
class DeMoStrategy(Strategy):
    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store DeMo-specific parameters
        self.compression_decay = compression_decay
        self.compression_topk = compression_topk
        self.compression_chunk = compression_chunk
        self.weight_decay = weight_decay

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        print("initialising DeMo engine")

        # Create DeMo optimizer with stored parameters
        demo_kwargs = {
            "compression_decay": self.compression_decay,
            "compression_topk": self.compression_topk,
            "compression_chunk": self.compression_chunk,
            "weight_decay": self.weight_decay,
            "custom_all_gather": all_gather,
        }

        # Add any additional optimizer kwargs from strategy config if they exist
        if hasattr(self, "strategy_config") and hasattr(
            self.strategy_config, "optimizer_kwargs"
        ):
            demo_kwargs.update(self.strategy_config.optimizer_kwargs)

        self.optim = DeMo(model.parameters(), **demo_kwargs)
        self._setup_scheduler()

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.optim.step()

        super().step()  # Print number of bytes communicated. This can be put in a different method tbh.


STRATEGY = DeMoStrategy(
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 1000,
        "cosine_anneal": True,
    },
)
