import math
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist
import torch.fft
from einops import rearrange
import datetime

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
def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None):
    return dist.all_reduce(tensor, op=op, group=group)


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
        return self.cls(model.named_parameters(), **(self.kwargs or {}))


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

# BASE COMPRESSION
class Quantizer(ABC):
    @abstractmethod
    def quantize(self, x: torch.Tensor):
        pass

    @abstractmethod
    def dequantize(self, x: torch.Tensor, meta):
        pass

    @abstractmethod
    def bits_per_value(self) -> int:
        pass

class UniformKBitQuantizer(Quantizer):
    def __init__(self, n_bins: int, range_in_sigmas: float):
        self.n_bins = n_bins
        # Check if n_bins is a power of 2
        if not (n_bins > 0 and (n_bins & (n_bins - 1) == 0)):
             raise ValueError("n_bins must be a power of 2 for k-bit quantization.")
        
        self.range = range_in_sigmas

    def bits_per_value(self):
        return int(math.log2(self.n_bins))

    @torch.no_grad()
    def quantize(self, val: torch.Tensor):
        offset = self.n_bins // 2
        
        # 1. Calculate Shift and Scale
        shift = val.mean()
        centered = val - shift

        if centered.numel() <= 1:
            std = torch.tensor(0.0, device=val.device)
        else:
            # We use the norm (sqrt(sum(x^2))) which approximates std * sqrt(N)
            std = centered.norm() / math.sqrt(centered.numel() - 1)

        scale = self.range * std / self.n_bins
        if scale.item() == 0 or not torch.isfinite(scale):
                    scale = torch.tensor(1.0, device=val.device)
        
        # 2. Quantization (Float -> Indices)
        q_indices = ((centered / scale) + offset).round().clamp(0, self.n_bins - 1).to(torch.uint8)

        # 3. Packing (The size reduction step)
        original_shape = q_indices.shape
        q_flat = q_indices.flatten()
        
        # Calculate padding needed to make the tensor length divisible by 4 (for 2-bit packing)
        packing_rate = 4  # 4 indices * 2 bits = 8 bits (1 byte)
        padding_len = (packing_rate - (q_flat.numel() % packing_rate)) % packing_rate
        
        if padding_len > 0:
            # Pad with zeros (index 0, which corresponds to the smallest magnitude)
            q_flat = torch.cat([q_flat, torch.zeros(padding_len, dtype=torch.uint8, device=q_flat.device)])

        # Reshape to (N/4, 4) where N is the padded length
        q_reshaped = q_flat.view(-1, packing_rate)
        
        # Pack 4 indices into 1 byte using bitwise shifts
        # Byte = [idx0] | [idx1 << 2] | [idx2 << 4] | [idx3 << 6]
        q_packed = (
            q_reshaped[:, 0] | 
            (q_reshaped[:, 1] << 2) | 
            (q_reshaped[:, 2] << 4) | 
            (q_reshaped[:, 3] << 6)
        )
        
        # Payload (q_packed) and Metadata (shift, scale, shape, padding)
        return q_packed, (shift, scale, original_shape, padding_len)

        # lookup = torch.zeros(self.n_bins, device=val.device)
        # lookup.scatter_add_(0, q.long().flatten(), centered.flatten())
        # counts = torch.zeros_like(lookup).scatter_add_(
        #     0, q.long().flatten(), torch.ones_like(centered.flatten())
        # )
        # lookup = torch.where(counts > 0, lookup / counts, 0.0)

        # return q, (shift, lookup, val.dtype)

    @torch.no_grad()
    def dequantize(self, q_packed: torch.Tensor, meta: Tuple[torch.Tensor, torch.Tensor, torch.Size, int]):
        # shift, lookup, dtype = meta
        # return (lookup[q.long()] + shift).to(dtype)
        shift, scale, shape, padding_len = meta
        offset = self.n_bins // 2
        
        # 1. UNPACKING
        mask = torch.tensor(3, dtype=torch.uint8, device=q_packed.device) # 0b11
        
        # Extract indices by shifting and masking
        # torch.stack creates a tensor of shape (4, N_packed)
        unpacked = torch.stack([
            (q_packed) & mask,
            (q_packed >> 2) & mask,
            (q_packed >> 4) & mask,
            (q_packed >> 6) & mask
        ], dim=0).transpose(0, 1).flatten() # (N_packed, 4) -> flatten
        
        # Remove padding
        if padding_len > 0:
            unpacked = unpacked[:-padding_len]
            
        # Reshape to original gradient shape
        q_indices = unpacked.view(shape)
        
        # 2. Dequantization (Indices -> Float)
        # Reconstruct the float value: (q - offset) * scale + shift
        # We must convert to float for the arithmetic
        return (q_indices.float() - offset) * scale + shift

class Sparsifier(ABC):
    @abstractmethod
    def sparsify(self, x: torch.Tensor):
        pass

    @abstractmethod
    def desparsify(self, payload, meta, ref: torch.Tensor):
        pass

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

class ChunkedTopKSparsifier(Sparsifier):
    def __init__(self, chunking: ChunkingTransform, k: int, random=False):
        self.chunking = chunking
        self.k = k
        self.random = random

    @torch.no_grad()
    def sparsify(self, x: torch.Tensor):
        x_c = self.chunking.encode(x)

        if x_c.ndim > 2:
            flat = rearrange(x_c, "y x h w -> y x (h w)")
        else:
            flat = x_c

        k = min(self.k, flat.shape[-1])
        if self.random:
            idx = torch.randint(flat.shape[-1], (k,), device=x.device)
        else:
            _, idx = torch.topk(flat.abs(), k, dim=-1, sorted=False)

        val = torch.gather(flat, -1, idx)
        return (idx, val), x_c.shape

    @torch.no_grad()
    def desparsify(self, payload, meta, ref: torch.Tensor):
        idx, val = payload
        shape = meta

        x = torch.zeros(shape, device=ref.device, dtype=ref.dtype)
        flat = rearrange(x, "y x h w -> y x (h w)") if x.ndim > 2 else x
        flat.scatter_reduce_(-1, idx, val, reduce="mean", include_self=False)

        return self.chunking.decode(x)


class Compression:
    def __init__(self, sparsifier: Optional[Sparsifier], quantizer: Optional[Quantizer]):
        self.sparsifier = sparsifier
        self.quantizer = quantizer

    def compress(self, x: torch.Tensor):

        if self.sparsifier:
            payload, s_meta = self.sparsifier.sparsify(x)
        else:
            payload, s_meta = x, None

        if self.quantizer:
            payload, q_meta = self.quantizer.quantize(payload)
        else:
            q_meta = None

        return payload, (s_meta, q_meta)

    def decompress(self, payload, meta, ref: torch.Tensor):
        s_meta, q_meta = meta

        if self.quantizer:
            payload = self.quantizer.dequantize(payload, q_meta)

        if self.sparsifier:
            payload = self.sparsifier.desparsify(payload, s_meta, ref)

        return payload

# DILOCO COMMUNICATOR
class DiLoCoCommunicator(CommunicationModule):
    """
    Communication module for master-worker setup (like DiLoCo).
    """

    def __init__(
        self,
        H: int = 100,
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        compression: Optional[Compression] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        self.H = H
        self.outer_optim_spec = ensure_optim_spec(
            outer_optim_spec,
            OptimSpec(torch.optim.SGD, lr=0.7, nesterov=True, momentum=0.9),
        )
        self.strategy = None  # Will be set by CommunicateOptimizeStrategy
        print(compression)
        self.process_group = process_group
        self.compression = compression
        self.error_buffers = {}
        self.master_model = None
        self.outer_optimizer = None

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform master-worker communication."""
        if num_nodes > 1 and local_step % self.H == 0 and local_step > 0:
            # # Original DiLoCo
            # # First average all models
            # for param in model.parameters():
            #     # all_reduce(param.data, op=dist.ReduceOp.SUM)
            #     # param.data /= num_nodes
                
            #     # Error feedback + Compression
            #     x = param.data + self.error_buffers[param]
            #     print(self.compression)
            #     if self.compression:
            #         payload, meta = self.compression.compress(x)
            #         all_reduce(payload, op=dist.ReduceOp.SUM)
            #         x_hat = self.compression.decompress(payload, meta, ref=param)
            #     else:
            #         all_reduce(x, op=dist.ReduceOp.SUM)
            #         x_hat = x

            #     x_hat /= num_nodes

            #     # Update error buffer
            #     self.error_buffers[param].copy_(x - x_hat)

            #     # Write back
            #     param.data.copy_(x_hat)

            # # Master does outer optimization step
            # if rank == 0 and self.master_model is not None:
            #     self.outer_optimizer.zero_grad()
            #     self._set_master_grad(model)
            #     self.outer_optimizer.step()
            #     self._synchronize_master_model(model)

            # # Broadcast updated parameters
            # for param in model.parameters():
            #     broadcast(param.data, src=0)

            # # Alternative DiLoCo with compression
            # if self.compression is not None:
            #         self.outer_optimizer.zero_grad()
            #         self._set_master_grad(model)

            #         for name, p in self.master_model.named_parameters():
            #             delta = p.grad
            #             delta_ef = delta + self.error_buffers[name]

            #             q, meta = self.compression.compress(delta_ef)
            #             delta_hat = self.compression.decompress(q, meta, ref=p)
            #             len(delta_hat)
            #             all_reduce(delta_hat, op=dist.ReduceOp.SUM, group=self.process_group)
            #             delta_hat /= num_nodes

            #             self.error_buffers[name].copy_(delta_ef - delta_hat)
            #             p.grad.copy_(delta_hat)              # what optimizer sees

            #         self.outer_optimizer.step()
            #         self._synchronize_master_model(model)
            
            # else:
                
            #     # Average worker parameters (DiLoCo step)
            #     for param in model.parameters():
            #         all_reduce(param.data, op=dist.ReduceOp.SUM, group=self.process_group)
            #         param.data /= num_nodes

            #     self.outer_optimizer.zero_grad()
            #     self._set_master_grad(model)
            #     self.outer_optimizer.step()
            #     self._synchronize_master_model(model)

            # New DiLoCo with All-Gather + Compression   
            if self.compression is not None:             
                # 1. Zero Outer Optimizer and Prepare Master Gradient (delta)
                self.outer_optimizer.zero_grad()
                self._set_master_grad(model)

                for name, p in self.master_model.named_parameters():
                    delta = p.grad # MasterParam - WorkerParam
                    
                    # 2. Apply Error Feedback and Compress (Locally)
                    delta_ef = delta + self.error_buffers[name]

                    # q_packed: The memory-reduced payload (e.g., 2-bit indices packed into uint8)
                    # meta: (shift, scale, original_shape, padding)
                    q_packed, meta = self.compression.compress(delta_ef)
                    # print(len(meta))
                    (s_meta, q_meta) = meta
                    
                    # Stack shift and scale into a tiny tensor for communication
                    shift, scale, shape, padding = q_meta
                    meta_payload = torch.stack([shift, scale])
                    
                    # --- 3. COMMUNICATION PHASE: All-Gather ---
                    # a) All-Gather Metadata (Crucial for correct decompression)
                    # We need N copies of the 2-float tensor
                    gathered_metas = [
                        torch.zeros_like(meta_payload) for _ in range(num_nodes)
                    ]
                    all_gather(gathered_metas, meta_payload, group=self.process_group)

                    # b) All-Gather Compressed Packed Data (Memory-Efficient Loop)
                    # We use a temporary buffer (instead of N buffers) and sum them up
                    delta_sum = torch.zeros_like(p) # The final aggregated update
                    
                    # List to hold all gathered packed tensors (must be done to use all_gather API)
                    gathered_q_list = [torch.empty_like(q_packed) for _ in range(num_nodes)]

                    # Perform the all-gather
                    all_gather(gathered_q_list, q_packed, group=self.process_group)

                    # --- 4. AGGREGATION & DECOMPRESSION (ON EVERY NODE) ---
                    # Iterate through all N gathered buffers (Memory intensive part, but unavoidable with all_gather)
                    for i in range(num_nodes):
                        node_q = gathered_q_list[i]
                        node_shift = gathered_metas[i][0]
                        node_scale = gathered_metas[i][1]
                        
                        # Reconstruct meta for node i (shape/padding is fixed for this tensor)
                        node_meta = (node_shift, node_scale, shape, padding)
                        
                        # Decompress back to full precision (on the local device)
                        delta_node_i = self.compression.decompress(node_q, (s_meta, node_meta), ref=p)
                        
                        # Accumulate the sum
                        delta_sum += delta_node_i
                        
                    delta_hat = delta_sum / num_nodes # The final averaged update

                    # --- 5. ERROR FEEDBACK AND OPTIMIZER STEP ---
                    # Calculate the error for the local node (using the locally quantized version)
                    my_delta_hat = self.compression.decompress(q_packed, meta, ref=p)
                    self.error_buffers[name].copy_(delta_ef - my_delta_hat)
                    
                    # The outer optimizer steps using the global average
                    p.grad.copy_(delta_hat)

                # Outer optimization and broadcast to workers
                self.outer_optimizer.step()
                self._synchronize_master_model(model)

            else:
                # Average worker parameters (DiLoCo step)
                for param in model.parameters():
                    all_reduce(param.data, op=dist.ReduceOp.SUM, group=self.process_group)
                    param.data /= num_nodes

                self.outer_optimizer.zero_grad()
                self._set_master_grad(model)
                self.outer_optimizer.step()
                self._synchronize_master_model(model)

            # # Alternative DiLoCo
            # self.outer_optimizer.zero_grad()
            # self._set_master_grad(model)
            # for param in model.parameters():
            #     all_reduce(param.data, op=dist.ReduceOp.SUM)
            #     param.data /= num_nodes
            # self.outer_optimizer.step()
            # self._synchronize_master_model(model)

            # # SparseLoCo
            # self.outer_optimizer.zero_grad()
            # self._set_master_grad(model)
            # self.outer_optimizer.step()
            # self._synchronize_master_model(model)

    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """Initialize master model for rank 0."""
        # # Original DiLoCo
        # if rank == 0:
        #     self.master_model = deepcopy(model).to("cpu")
        #     for param in self.master_model.parameters():
        #         param.requires_grad = True
        #     self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

        self.process_group = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(60)
        )
        self.master_model = deepcopy(model).to("cpu")
        for param in self.master_model.parameters():
            param.requires_grad = True

        # build outer optimizer with the Gloo group
        self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

        # EF buffers exist on all ranks
        self.error_buffers = {}
        for name, p in self.master_model.named_parameters():
            self.error_buffers[name] = torch.zeros_like(p.data)

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
        compression: Optional[Compression] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        H: int = 100,
        **kwargs,
    ):
        self.H = H

        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(optim_spec, OptimSpec(torch.optim.AdamW))

        # Create the DiLoCo communicator
        self.diloco_comm = DiLoCoCommunicator(
            H=H, 
            outer_optim_spec=outer_optim_spec,
            compression=compression,
            process_group=process_group,
        )

        super().__init__(
            optim_spec=optim_spec, communication_modules=[self.diloco_comm], **kwargs
        )


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, params):
        params = list(params)
        parameters_muon = [
            p
            for n, p in params
            if (p.ndim >= 2) and ("embed" not in n) and ("head" not in n)
        ]
        parameters_adam = [
            p for n, p in params if (p.ndim < 2) or ("embed" in n) or ("head" in n)
        ]
        print(f"LENGTH MUON {len(parameters_muon)}")
        print(f"LENGTH ADAM {len(parameters_adam)}")
        param_groups = [
            dict(
                params=parameters_muon,
                use_muon=True,
                lr=2.6827e-2,
                weight_decay=0.01,
            ),
            dict(
                params=parameters_adam,
                use_muon=False,
                lr=0.001,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            ),
        ]
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


STRATEGY = DiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001),
    compression = Compression(
        sparsifier=None,
        quantizer=UniformKBitQuantizer(n_bins=4, range_in_sigmas=6),
    ),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 500,
        "cosine_anneal": True,
    },
    max_norm=1.0,
    H=15,
)
