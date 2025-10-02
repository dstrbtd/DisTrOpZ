# exogym/aux/comm_track.py
import threading, torch
import torch.distributed as dist

_comm = threading.local()


def _init_counters():
    _comm.bytes = 0
    _comm.by_op = {}


def _bump(op, tensor):
    n = tensor.numel() * tensor.element_size()
    _comm.bytes += n
    _comm.by_op[op] = _comm.by_op.get(op, 0) + n


def patch_collectives():
    _init_counters()

    def wrap1(name, fn):
        def inner(t, *a, **k):
            _bump(name, t)
            return fn(t, *a, **k)

        return inner

    def wrap_allgather(name, fn):
        def inner(out_list, t, *a, **k):
            _bump(name, t)
            return fn(out_list, t, *a, **k)

        return inner

    dist.all_reduce = wrap1("all_reduce", dist.all_reduce)
    dist.reduce = wrap1("reduce", dist.reduce)
    dist.broadcast = wrap1("broadcast", dist.broadcast)
    dist.all_gather = wrap_allgather("all_gather", dist.all_gather)
    # add scatter/gather/all_to_all if you use them


def comm_bytes():
    return getattr(_comm, "bytes", 0), getattr(_comm, "by_op", {})
