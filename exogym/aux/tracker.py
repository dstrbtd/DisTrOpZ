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

    def wrap_base(name, fn):
        def inner(t, *a, **k):
            _bump(name, t)
            return fn(t, *a, **k)

        return inner

    def wrap_scatter(name, fn):
        def inner(output, scatter_list=None, src=0, *a, **k):
            if scatter_list is not None:
                # root rank
                size = sum(t.numel() * t.element_size() for t in scatter_list)
            else:
                # non-root
                size = output.numel() * output.element_size()
            _comm.bytes += size
            _comm.by_op[name] = _comm.by_op.get(name, 0) + size
            return fn(output, scatter_list, src, *a, **k)
        return inner

    def wrap_allgather(name, fn):
        def inner(out_list, t, *a, **k):
            _bump(name, t)
            return fn(out_list, t, *a, **k)

        return inner
    
    def wrap_alltoall(name, fn):
        def inner(output, input, *a, **k):
            size = input.numel() * input.element_size()
            _comm.bytes += size
            _comm.by_op[name] = _comm.by_op.get(name, 0) + size
            return fn(output, input, *a, **k)
        return inner

    dist.all_reduce = wrap_base("all_reduce", dist.all_reduce)
    dist.reduce = wrap_base("reduce", dist.reduce)
    dist.broadcast = wrap_base("broadcast", dist.broadcast)
    dist.all_gather = wrap_allgather("all_gather", dist.all_gather)
    dist.scatter = wrap_scatter("scatter", dist.scatter)
    dist.all_to_all = wrap_alltoall("all_to_all", dist.all_to_all)


def comm_bytes():
    return getattr(_comm, "bytes", 0), getattr(_comm, "by_op", {})
