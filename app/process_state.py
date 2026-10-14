"""Conservative PID inspection without optional packages or Windows kill(0)."""
from __future__ import annotations

import os


def process_exists(pid: int) -> bool:
    if type(pid) is not int or pid < 1:
        raise ValueError("PID must be a positive integer")
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        kernel.CloseHandle(handle)
        return True  # Includes recently exited/unreaped owners; fail closed.
    error = ctypes.get_last_error()
    if error == 87:  # ERROR_INVALID_PARAMETER: no process with this PID.
        return False
    if error == 5:  # Access denied is not evidence of process death.
        return True
    raise OSError(error, "cannot verify workflow lock owner")
