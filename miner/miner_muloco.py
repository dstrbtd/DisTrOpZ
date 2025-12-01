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
            # Original DiLoCo
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
        # Original DiLoCo
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


import torch
import torch.distributed as dist


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
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
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
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, params):
        # breakpoint()
        print([p for p in params])
        lr = 3e-4
        parameters_muon = [
            p
            for n, p in params
            if (p.ndim >= 2) and ("embed" not in n) and ("head" not in n)
        ]
        parameters_adam = [
            p
            for n, p in params
            if (p.ndim < 2) or ("embed" in n) or ("head" in n)
        ]
        param_groups = [
            dict(
                params=parameters_muon,
                use_muon=True,
                lr=0.02,
                weight_decay=0.01,
            ),
            dict(
                params=parameters_adam,
                use_muon=False,
                lr=lr,
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
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
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
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

STRATEGY = DiLoCoStrategy(
    optim_spec=OptimSpec(SingleDeviceMuonWithAuxAdam),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 500,
        "cosine_anneal": True,
    },
    max_norm=1.0,
    H=15,
)
