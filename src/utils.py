"""Process helpers: clean exit under SLURM, one process per chunk, multi-GPU parameter averaging."""
import os
import signal
import subprocess
import sys
import threading


def exit_cleanly_on_sigterm():
    """Exit via ``os._exit`` on SIGTERM/SIGINT (e.g. SLURM preemption); call before creating the simulator."""
    sigs = {signal.SIGTERM, signal.SIGINT}
    hook = [None]
    signal.pthread_sigmask(signal.SIG_BLOCK, sigs)

    def wait():
        sig = signal.sigwait(sigs)
        if hook[0] is not None:
            try:
                hook[0]()
            except Exception:
                pass
        os._exit(128 + int(sig))   # Non-zero so that schedulers and `set -e` see the interruption
    threading.Thread(target=wait, daemon=True).start()
    return hook


def hard_exit(code=0):
    """Exit immediately, skipping interpreter teardown (see ``exit_cleanly_on_sigterm``)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run_each(commands):
    """Run every command in its own process (one simulator per process); return the failed ones."""
    failed = []
    for cmd in commands:
        cmd = [str(c) for c in cmd]
        print("[run] " + " ".join(cmd), flush=True)
        # children must not inherit the signal mask of exit_cleanly_on_sigterm (torchrun/srun forward signals)
        unblock = lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})  # noqa: E731
        if subprocess.call(cmd, preexec_fn=unblock) != 0:
            print("[run] FAILED: " + " ".join(cmd), file=sys.stderr, flush=True)
            failed.append(cmd)
    return failed


# ---------------------------------------------------------------- Multi-GPU (parameter averaging)
# Parameters are averaged across ranks after each iteration; torch is imported lazily (before isaacgym).

def init_distributed():
    """Return ``(rank, world_size, local_rank)`` from torchrun or SLURM variables; start NCCL if world > 1."""
    import torch
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29513")
        local = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, local


def average_parameters(params, world):
    """Average ``params`` in place across ranks."""
    if world <= 1:
        return
    import torch
    import torch.distributed as dist
    with torch.no_grad():
        for p in params:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= world


def average_scalar(x, world, device):
    """Average the scalar ``x`` across ranks."""
    if world <= 1:
        return x
    import torch
    import torch.distributed as dist
    with torch.no_grad():
        t = torch.tensor([float(x)], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / world).item()


def barrier(world):
    """Synchronize all ranks (no-op for a single process)."""
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
