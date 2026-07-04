from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])


class SystemStatus(BaseModel):
    gpu_available: bool
    gpu_name: str
    gpu_memory_used_mb: int | None
    gpu_memory_total_mb: int | None
    cpu_memory_used_mb: int | None
    cpu_memory_total_mb: int | None
    loaded_models: list[str]
    current_model: str | None


def _get_gpu_memory():
    """Return (used_mb, total_mb) for CUDA or DirectML."""
    # CUDA path
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            used = torch.cuda.memory_allocated(0) // (1024 * 1024)
            return used, total
    except Exception:
        pass

    # DirectML path — fall back to Win32 system memory via ctypes
    try:
        import torch_directml
    except ImportError:
        pass

    # Win32 GlobalMemoryStatusEx
    try:
        import ctypes, ctypes.wintypes
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.wintypes.ULONGLONG),
                ("ullAvailPhys", ctypes.wintypes.ULONGLONG),
                ("ullTotalPageFile", ctypes.wintypes.ULONGLONG),
                ("ullAvailPageFile", ctypes.wintypes.ULONGLONG),
                ("ullTotalVirtual", ctypes.wintypes.ULONGLONG),
                ("ullAvailVirtual", ctypes.wintypes.ULONGLONG),
                ("ullAvailExtendedVirtual", ctypes.wintypes.ULONGLONG),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_mb = stat.ullTotalPhys // (1024 * 1024)
        used_mb = total_mb - (stat.ullAvailPhys // (1024 * 1024))
        return used_mb, total_mb
    except Exception:
        return None, None


def _get_cpu_memory():
    """Return (used_mb, total_mb) via psutil or Win32 fallback."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used // (1024 * 1024), vm.total // (1024 * 1024)
    except ImportError:
        pass

    # Win32 fallback (same as gpu on machines without CUDA)
    try:
        import ctypes, ctypes.wintypes
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.wintypes.ULONGLONG),
                ("ullAvailPhys", ctypes.wintypes.ULONGLONG),
                ("ullTotalPageFile", ctypes.wintypes.ULONGLONG),
                ("ullAvailPageFile", ctypes.wintypes.ULONGLONG),
                ("ullTotalVirtual", ctypes.wintypes.ULONGLONG),
                ("ullAvailVirtual", ctypes.wintypes.ULONGLONG),
                ("ullAvailExtendedVirtual", ctypes.wintypes.ULONGLONG),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        used = stat.ullTotalPhys - stat.ullAvailPhys
        total = stat.ullTotalPhys
        return used // (1024 * 1024), total // (1024 * 1024)
    except Exception:
        return None, None


@router.get("/status", response_model=SystemStatus)
async def get_status():
    from app.services.generator import get_loaded_model, get_loaded_models

    gpu_name = "CPU"
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_available = True
        else:
            try:
                import torch_directml
                gpu_name = "AMD GPU (DirectML)"
                gpu_available = True
            except ImportError:
                gpu_name = "CPU"
                gpu_available = False
    except Exception:
        pass

    gpu_used, gpu_total = _get_gpu_memory()
    cpu_used, cpu_total = _get_cpu_memory()

    return SystemStatus(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
        cpu_memory_used_mb=cpu_used,
        cpu_memory_total_mb=cpu_total,
        loaded_models=get_loaded_models(),
        current_model=get_loaded_model(),
    )
