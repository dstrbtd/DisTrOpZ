import math
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist
import torch.fft
from einops import rearrange

from copy import deepcopy
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from typing import (
    List,
    Type,
    Union,
    Optional,
    Dict,
    Any,
    TypeAlias,
    Callable,
    Iterable,
    Tuple,
)
from abc import ABC, abstractmethod

from exogym.aux.utils import LogModule

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


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


# DILOCO COMMUNICATOR
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


# DILOCO STRATEGY
class DiLoCoStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        optim_spec: Optional[
            Union[str, OptimSpec]
        ] = None,  # inner optimizer is named optim_spec for consistency
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        H: int = 100,
        **kwargs,
    ):
        self.H = H

        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(optim_spec, OptimSpec(torch.optim.AdamW))

        # Create the DiLoCo communicator
        self.diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)

        super().__init__(
            optim_spec=optim_spec, communication_modules=[self.diloco_comm], **kwargs
        )


# https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
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


class ChunkingTransform:
    """Handles tensor chunking, with an optional DCT for DeMo reproduction."""

    def __init__(
        self, param_groups: ParamsT, chunk_size: int, use_dct: bool, norm: str = "ortho"
    ):
        self.target_chunk = chunk_size
        self.use_dct = use_dct
        self.decode_info: Optional[Tuple[str, torch.Size]] = None
        self.shape_dict = {}
        self.f_dict, self.b_dict = {}, {}
        self._initialize_transforms(param_groups, norm)

    def _initialize_transforms(self, param_groups: ParamsT, norm: str):
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    if s not in self.shape_dict:
                        sc = _get_smaller_split(s, self.target_chunk)
                        self.shape_dict[s] = sc
                        if self.use_dct and sc not in self.f_dict:
                            I = torch.eye(sc, device=p.device, dtype=p.dtype)
                            self.f_dict[sc] = _dct(I, norm=norm)
                            self.b_dict[sc] = _idct(I, norm=norm)

    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            return torch.einsum("...ijkl, kb, ld -> ...ijbd", x, b, d)

    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            return torch.einsum("...ijbd, bk, dl -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 1:
            self.decode_info = ("1d", x.shape)
            n1 = self.shape_dict[x.shape[0]]
            x_chunked = rearrange(x, "(c s) -> c s", s=n1)
            if not self.use_dct:
                return x_chunked
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w
            return self.einsum_2d(x_chunked, n1w)

        self.decode_info = ("2d", x.shape)
        n1 = self.shape_dict[x.shape[0]]
        n2 = self.shape_dict[x.shape[1]]
        x_chunked = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
        if not self.use_dct:
            return x_chunked
        n1w = self.f_dict[n1].to(x.device)
        n2w = self.f_dict[n2].to(x.device)
        self.f_dict[n1] = n1w
        self.f_dict[n2] = n2w
        return self.einsum_2d(x_chunked, n1w, n2w)

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.decode_info is None:
            raise RuntimeError("decode() called before encode()")
        strategy, _ = self.decode_info

        if strategy == "1d":
            if self.use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(x.device)
                self.b_dict[n1] = n1w
                x = self.einsum_2d_t(x, n1w)
            return rearrange(x, "c s -> (c s)")

        if self.use_dct:
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w
            x = self.einsum_2d_t(x, n1w, n2w)
        return rearrange(x, "y x h w -> (y h) (x w)")


class TopKCompressor:
    """Top-K / Random-K sparsification with optional statistical quantization."""

    def __init__(
        self,
        use_quantization: bool,
        n_bins: int,
        range_in_sigmas: float,
        use_randomk: bool = False,
    ):
        self.use_quantization = use_quantization
        self.use_randomk = use_randomk
        self.rng = None
        if self.use_randomk:
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.rng = torch.Generator(device=f"cuda:{rank}")
            self.rng.manual_seed(42 + rank)

        if use_quantization:
            self.n_bins = n_bins
            self.range_in_sigmas = range_in_sigmas

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x: torch.Tensor, k: int):
        if x.ndim > 2:
            x_flat_chunks = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat_chunks = rearrange(x, "x w -> x w")

        k = self._clamp_topk(x_flat_chunks, k)

        if self.use_randomk:
            rand_vals = torch.empty_like(x_flat_chunks).uniform_(
                0.0, 1.0, generator=self.rng
            )
            _, idx = torch.topk(rand_vals, k=k, dim=-1)
        else:
            _, idx = torch.topk(
                x_flat_chunks.abs(), k=k, dim=-1, largest=True, sorted=False
            )

        val = torch.gather(x_flat_chunks, dim=-1, index=idx)

        quant_params = None
        if self.use_quantization:
            quantized_val, quant_params = self._quantize(val)
            val = quantized_val

        return idx.to(torch.int64), val, x.shape, quant_params

    @torch.no_grad()
    def decompress(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        x_shape: Tuple,
        ref_param: torch.Tensor,
        quant_params: Optional[Tuple],
    ) -> torch.Tensor:
        if quant_params is not None:
            val = self._dequantize(val, quant_params)

        x = torch.zeros(x_shape, device=ref_param.device, dtype=ref_param.dtype)
        if len(x_shape) > 2:
            x_flat = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat = x

        x_flat.scatter_reduce_(
            dim=-1,
            index=idx.to(torch.int64),
            src=val.to(ref_param.dtype),
            reduce="mean",
            include_self=False,
        )

        if len(x_shape) > 2:
            x = rearrange(x_flat, "y x (h w) -> y x h w", h=x_shape[2])
        else:
            x = x_flat
        return x

    @torch.no_grad()
    def batch_decompress(
        self, idx_list: list, val_list: list, x_shape: Tuple, ref_param: torch.Tensor
    ) -> torch.Tensor:
        idx_all = torch.cat([i.to(ref_param.device) for i in idx_list], dim=-1)
        val_all = torch.cat(
            [v.to(ref_param.device, ref_param.dtype) for v in val_list], dim=-1
        )

        x = torch.zeros(x_shape, device=ref_param.device, dtype=ref_param.dtype)
        if len(x_shape) > 2:
            x_flat = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat = x

        x_flat.scatter_reduce_(
            dim=-1,
            index=idx_all.to(torch.int64),
            src=val_all,
            reduce="mean",
            include_self=False,
        )

        if len(x_shape) > 2:
            x = rearrange(x_flat, "y x (h w) -> y x h w", h=x_shape[2])
        else:
            x = x_flat
        return x

    def _quantize(self, val: torch.Tensor):
        offset = self.n_bins // 2
        shift = val.mean()
        centered_val = val - shift

        if centered_val.numel() <= 1:
            std_unbiased = torch.tensor(0.0, device=val.device, dtype=val.dtype)
        else:
            std_unbiased = centered_val.norm() / math.sqrt(centered_val.numel() - 1)

        scale = self.range_in_sigmas * std_unbiased / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered_val.dtype, device=val.device)

        quantized = (
            (centered_val.float() / scale + offset).round().clamp(0, self.n_bins - 1)
        ).to(torch.uint8)

        lookup = torch.zeros(self.n_bins, dtype=torch.float32, device=val.device)
        sums = torch.zeros_like(lookup).scatter_add_(
            0, quantized.long().flatten(), centered_val.float().flatten()
        )
        counts = torch.zeros_like(lookup).scatter_add_(
            0,
            quantized.long().flatten(),
            torch.ones_like(centered_val.float().flatten()),
        )
        lookup = torch.where(counts > 0, sums / counts, 0.0)

        params_tuple = (shift, float(scale), offset, lookup, val.dtype)
        return quantized, params_tuple

    def _dequantize(self, val: torch.Tensor, quant_params: Tuple):
        if quant_params is None:
            return val
        shift, _, _, lookup, orig_dtype = quant_params
        dequantized = lookup.to(val.device)[val.long()] + shift
        return dequantized.to(orig_dtype)


class SparseLoCo(torch.optim.SGD):
    """Implements the SparseLoCo optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        error_decay: float = 0.999,
        top_k: int = 32,
        use_randomk: bool = False,
        chunk_size: int = 64,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        use_dct: bool = False,
        use_sign: bool = False,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=0.0,
            **kwargs,
        )

        self.error_decay = error_decay
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.decoupled_weight_decay = weight_decay
        self.use_dct = use_dct
        self.use_sign = use_sign
        self.process_group = process_group

        self.chunking = ChunkingTransform(self.param_groups, chunk_size, use_dct)
        self.compressor = TopKCompressor(
            use_quantization,
            quantization_bins,
            quantization_range,
            use_randomk=use_randomk,
        )

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["error_buffer"] = torch.zeros_like(p)

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        ws = dist.get_world_size(self.process_group)
        tensor_list = [torch.zeros_like(tensor) for _ in range(ws)]
        dist.all_gather(tensor_list, tensor, group=self.process_group)
        return tensor_list

    def _all_gather_quant_params(self, quant_params: Tuple) -> List[Tuple]:
        (shift, scale, offset, lookup, dtype) = quant_params
        comm_tensor = torch.cat([shift.view(1), lookup.to(shift.device)])
        comm_list = self._all_gather_tensor(comm_tensor)
        return [(t[0].unsqueeze(0), scale, offset, t[1:], dtype) for t in comm_list]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        if closure:
            closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.decoupled_weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self.decoupled_weight_decay)

                state = self.state[p]
                error_buffer = state["error_buffer"]

                if self.error_decay != 1.0:
                    error_buffer.mul_(self.error_decay)
                error_buffer.add_(p.grad, alpha=lr)

                tensor_to_compress = self.chunking.encode(error_buffer)

                k = self.top_k
                if not self.use_dct:
                    if tensor_to_compress.ndim > 2:
                        h = tensor_to_compress.shape[-2]
                        w = tensor_to_compress.shape[-1]
                        chunk_len = h * w
                    else:
                        chunk_len = tensor_to_compress.shape[-1]
                    k = int(
                        max(1, round(chunk_len * self.top_k / (self.chunk_size**2)))
                    )

                indices, values, shape, local_quant_params = self.compressor.compress(
                    tensor_to_compress, k
                )

                local_reconstruction = self.compressor.decompress(
                    indices, values, shape, p, local_quant_params
                )
                transmitted_gradient = self.chunking.decode(local_reconstruction)
                error_buffer.sub_(transmitted_gradient)

                gathered_quant_params = (
                    self._all_gather_quant_params(local_quant_params)
                    if self.compressor.use_quantization
                    and local_quant_params is not None
                    else None
                )
                gathered_indices = self._all_gather_tensor(indices)
                gathered_values = self._all_gather_tensor(values)

                if self.compressor.use_quantization and gathered_quant_params:
                    gathered_values = [
                        self.compressor._dequantize(v, qp)
                        for v, qp in zip(gathered_values, gathered_quant_params)
                    ]

                aggregated_reconstruction = self.compressor.batch_decompress(
                    gathered_indices, gathered_values, shape, p
                )
                aggregated_gradient = self.chunking.decode(aggregated_reconstruction)

                p.grad.copy_(aggregated_gradient)
                if self.use_sign:
                    p.grad.sign_()

        super().step()


STRATEGY = DiLoCoStrategy(
    optim_spec=OptimSpec(SparseLoCo, lr=0.0004),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 1000,
        "cosine_anneal": True,
    },
    max_norm=1.0,
    H=5,
)
