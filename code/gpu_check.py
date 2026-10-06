#!/usr/bin/env python3
"""Deployment verification: GPU check.

Prints the Code Ocean identity banner, what nvidia-smi reports (GPU name, driver, memory) and
a PyTorch CUDA check that runs a small matrix multiply on the GPU and compares it with the
same multiply on the CPU. Writes <results>/gpu_check.json and <results>/result.txt.

Exit 0 only when nvidia-smi sees a GPU, CUDA is available to PyTorch and the GPU result
matches the CPU result. Otherwise exit 1, with the reason in the log and result.txt.
Use a PyTorch (CUDA) starter environment. No network access; the only environment values
printed are CO_CAPSULE_ID, CO_COMPUTATION_ID, CO_CPUS and CO_MEMORY.
"""
import json
import os
import shutil
import subprocess
import sys
import time

IDENTITY = {"CO_CAPSULE_ID": "capsule", "CO_COMPUTATION_ID": "computation",
            "CO_CPUS": "cpus", "CO_MEMORY": "memory"}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, "nvidia-smi not found"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, e.__class__.__name__
    if out.returncode != 0:
        return None, (out.stderr or out.stdout).strip()[:300]
    return [line.strip() for line in out.stdout.splitlines() if line.strip()], None


def torch_check():
    try:
        import torch
    except ImportError:
        return {"torch": None, "error": "PyTorch is not installed: use a PyTorch (CUDA) starter environment"}
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count())}
    if not info["cuda_available"]:
        info["error"] = "torch.cuda.is_available() is False"
        return info
    info["devices"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256) / 1000.0
    b = a.t().contiguous()
    on_cpu = a @ b
    on_gpu = (a.cuda() @ b.cuda()).cpu()
    diff = float((on_cpu - on_gpu).abs().max())
    scale = float(on_cpu.abs().max())
    info["matmul_max_abs_diff"] = diff
    info["matmul_ok"] = diff <= 1e-4 * scale
    if not info["matmul_ok"]:
        info["error"] = f"GPU matmul differs from CPU by {diff} (scale {scale})"
    return info


def main():
    results = os.environ.get("DV_RESULTS_DIR", "/results")
    identity = {k: os.environ.get(k, "unset") for k in IDENTITY}
    print("DV-BANNER " + " ".join(f"{IDENTITY[k]}={v}" for k, v in identity.items()), flush=True)

    gpus, smi_error = nvidia_smi()
    if gpus:
        for line in gpus:
            print(f"DV-GPU {line}", flush=True)
    else:
        print(f"DV-GPU none ({smi_error})", flush=True)

    t = torch_check()
    print("DV-TORCH " + " ".join(f"{k}={v}" for k, v in t.items()), flush=True)

    ok = bool(gpus) and t.get("cuda_available") is True and t.get("matmul_ok") is True
    reason = "" if ok else (smi_error if not gpus else t.get("error", "unknown"))
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "gpu_check.json"), "w") as f:
        json.dump({"utc": utc(), "identity": identity, "nvidia_smi": gpus,
                   "nvidia_smi_error": smi_error, "torch": t, "ok": ok}, f, indent=2)
        f.write("\n")
    status = "OK" if ok else "FAIL"
    with open(os.path.join(results, "result.txt"), "w") as f:
        f.write(f"DV-GPU-CHECK {status} {utc()}" + (f" reason={reason}" if reason else "") + "\n")
    print(f"DV-GPU-CHECK {status}" + (f" reason={reason}" if reason else ""), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
