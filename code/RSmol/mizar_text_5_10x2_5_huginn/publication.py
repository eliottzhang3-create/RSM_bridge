"""Isolated no-overwrite publisher, copied from the existing Mizar converter."""
import ctypes
import errno
import os
import sys
from pathlib import Path


def _linux_noreplace(staging, output):
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise OSError(errno.ENOSYS, "libc has no renameat2", str(output))
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _reserved_posix_rename(staging, output):
    """Claim a NEW empty destination, then atomically replace our own reservation.

    POSIX rename needs no renameat2 flags. mkdir rejects existing destinations,
    including empty directories. Readers see either an empty reservation (which
    fails artifact validation) or the entire verified artifact. Never remove a
    nonempty or replaced reservation when handling failure.
    """
    output.mkdir(mode=0o700)
    owner = output.lstat()

    def still_owned():
        try:
            current = output.lstat()
            return (not output.is_symlink()
                    and (current.st_dev, current.st_ino) == (owner.st_dev, owner.st_ino))
        except FileNotFoundError:
            return False

    try:
        if not still_owned():
            raise FileExistsError(f"publication reservation was replaced: {output}")
        os.rename(staging, output)
    except BaseException:
        if still_owned():
            try:
                output.rmdir()  # Empty, owned reservation only; no recursive delete.
            except OSError:
                pass
        raise


def atomic_publish(staging, output):
    """Install sibling directories; existing outputs are always rejected."""
    staging, output = Path(staging).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output = output.parent.resolve() / output.name
    if not staging.is_dir() or staging.parent != output.parent:
        raise ValueError("publication requires a staging directory beside its output")
    if sys.platform.startswith("linux"):
        try:
            _linux_noreplace(staging, output)
            return "linux_renameat2_noreplace"
        except OSError as error:
            unsupported = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}
            if error.errno not in unsupported:
                raise
        _reserved_posix_rename(staging, output)
        return "posix_rename_into_exclusive_empty_reservation"
    elif sys.platform == "win32":
        os.rename(staging, output)  # Windows rename rejects an existing destination.
        return "windows_rename"
    else:
        raise RuntimeError("atomic no-replace publication currently supports Linux and Windows")


