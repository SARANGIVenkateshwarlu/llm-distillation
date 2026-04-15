"""
Environment and hardware utility functions.
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import torch


def get_device(preferred_device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        preferred_device: "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    Returns:
        torch.device object
    
    Example:
        >>> device = get_device("auto")
        >>> print(device)
        cuda:0
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preferred_device)
    
    return device


def get_gpu_info() -> Dict[str, any]:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
            info["devices"].append(device_info)
    
    return info


def check_gpu_memory(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Check GPU memory usage.
    
    Args:
        device: Device to check. If None, checks current device.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    if device is None:
        device = torch.cuda.current_device()
    else:
        device = device.index if device.type == "cuda" else 0
    
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free = total - reserved
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free,
        "utilization_percent": (allocated / total) * 100
    }


def print_gpu_memory(device: Optional[torch.device] = None) -> None:
    """
    Print GPU memory usage in a readable format.
    
    Args:
        device: Device to check. If None, prints all devices.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    if device is not None:
        devices = [device]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    
    for dev in devices:
        mem = check_gpu_memory(dev)
        print(f"GPU {dev}:")
        print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {mem['reserved_gb']:.2f} GB")
        print(f"  Total:     {mem['total_gb']:.2f} GB")
        print(f"  Free:      {mem['free_gb']:.2f} GB")


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_compute_capability(device: Optional[int] = None) -> Tuple[int, int]:
    """
    Get CUDA compute capability for a device.
    
    Args:
        device: Device index. If None, uses current device.
    
    Returns:
        Tuple of (major, minor) compute capability
    """
    if not torch.cuda.is_available():
        return (0, 0)
    
    if device is None:
        device = torch.cuda.current_device()
    
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def supports_bf16(device: Optional[int] = None) -> bool:
    """
    Check if device supports bfloat16 (Ampere architecture or newer).
    
    Args:
        device: Device index. If None, checks current device.
    
    Returns:
        True if bfloat16 is supported
    """
    major, _ = get_compute_capability(device)
    return major >= 8  # Ampere (SM 8.0) and newer


def get_optimal_dtype(device: Optional[int] = None) -> torch.dtype:
    """
    Get the optimal dtype for the device.
    
    Returns bfloat16 if supported, otherwise float16.
    
    Args:
        device: Device index. If None, uses current device.
    
    Returns:
        Optimal torch dtype
    """
    if supports_bf16(device):
        return torch.bfloat16
    else:
        return torch.float16


def setup_mixed_precision(precision: str = "auto") -> Tuple[torch.dtype, bool]:
    """
    Setup mixed precision training configuration.
    
    Args:
        precision: "auto", "bf16", "fp16", or "fp32"
    
    Returns:
        Tuple of (dtype, use_amp)
    """
    if precision == "auto":
        if supports_bf16():
            dtype = torch.bfloat16
            use_amp = True
        elif torch.cuda.is_available():
            dtype = torch.float16
            use_amp = True
        else:
            dtype = torch.float32
            use_amp = False
    elif precision == "bf16":
        if not supports_bf16():
            warnings.warn("bfloat16 not supported on this device, falling back to float16")
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        use_amp = torch.cuda.is_available()
    elif precision == "fp16":
        dtype = torch.float16
        use_amp = torch.cuda.is_available()
    else:  # fp32
        dtype = torch.float32
        use_amp = False
    
    return dtype, use_amp


def get_hardware_summary() -> str:
    """
    Get a summary of the hardware configuration.
    
    Returns:
        Formatted string with hardware information
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Hardware Summary")
    lines.append("=" * 50)
    
    # PyTorch version
    lines.append(f"PyTorch version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        lines.append(f"CUDA available: Yes")
        lines.append(f"CUDA version: {torch.version.cuda}")
        lines.append(f"Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            lines.append(f"  Compute Capability: {props.major}.{props.minor}")
            lines.append(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            lines.append(f"  Multi-Processor Count: {props.multi_processor_count}")
            lines.append(f"  BF16 Support: {'Yes' if props.major >= 8 else 'No'}")
    else:
        lines.append("CUDA available: No")
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        lines.append("MPS (Apple Silicon): Available")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def print_hardware_summary() -> None:
    """Print hardware summary."""
    print(get_hardware_summary())


def set_cuda_visible_devices(devices: str) -> None:
    """
    Set visible CUDA devices.
    
    Args:
        devices: Comma-separated string of device indices (e.g., "0,1")
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
