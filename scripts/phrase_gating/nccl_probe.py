"""Minimal four-rank all-reduce: is NCCL usable on these cards at all?"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
t = torch.ones(1024, 1024, device="cuda") * (rank + 1)
dist.all_reduce(t)
expected = sum(range(1, world + 1))
print(f"rank {rank}/{world} on {torch.cuda.get_device_name()} -> {t[0,0].item():.0f} "
      f"(expected {expected})", flush=True)
dist.barrier()
if rank == 0:
    print("NCCL_PROBE_OK", flush=True)
dist.destroy_process_group()
