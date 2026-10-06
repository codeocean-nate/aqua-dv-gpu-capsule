# aqua-dv-gpu-capsule

A small capsule used by the **deployment-verification** Aqua skill
([aqua-dv-skill](https://github.com/codeocean-nate/aqua-dv-skill)) to check that GPU runs work
on a Code Ocean deployment.

Use a **PyTorch (CUDA) starter environment**. The capsule makes no network calls.

## What a run does

`code/run` calls `code/gpu_check.py`, which:

1. prints the Code Ocean identity banner (`CO_CAPSULE_ID`, `CO_COMPUTATION_ID`, `CO_CPUS`, `CO_MEMORY`);
2. prints the GPU name, driver and memory from `nvidia-smi`;
3. checks that PyTorch sees CUDA, then runs a small matrix multiply on the GPU and compares it
   with the same multiply on the CPU;
4. writes `/results/gpu_check.json` and `/results/result.txt`.

| Exit code | Meaning |
|---|---|
| 0 | `DV-GPU-CHECK OK`: a GPU is visible and CUDA works |
| 1 | `DV-GPU-CHECK FAIL`, with the reason (no GPU, no CUDA, PyTorch missing, or a wrong result) |

A failing run is a real failure: it means the GPU isn't usable from a capsule on this deployment.

## Make a capsule from it

In Code Ocean: **New Capsule > Copy from Git**, paste
`https://github.com/codeocean-nate/aqua-dv-gpu-capsule`, then choose a PyTorch GPU starter
environment and a GPU machine. Or ask Aqua, with the deployment-verification skill enabled.
