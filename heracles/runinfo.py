"""Run bookkeeping: git commit of the code, peak memory, per-fit log handlers, device choice."""

import logging
import os
import platform
import subprocess

import torch

try:
    import resource
except ImportError:          # not a Unix platform
    resource = None


__all__ = [
    "git_commit",
    "peak_rss_mb",
    "setup_run_logging",
    "teardown_run_logging",
    "resolve_device",
]


def _git(args, cwd):
    return subprocess.check_output(["git"] + args, cwd=cwd,
                                   stderr=subprocess.DEVNULL).decode().strip()


def git_commit(repo_dir=None):
    """Returns the short git hash of a heracles source checkout, or None.

    None for an installed package (no checkout of heracles contains it) or
    without git.  The hash ends in ``"+dirty"`` if the working tree has
    uncommitted changes.
    """
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        top = _git(["rev-parse", "--show-toplevel"], repo_dir)
        if not repo_dir.startswith(os.path.join(top, "heracles")):
            return None
        commit = _git(["rev-parse", "--short", "HEAD"], repo_dir)
        if _git(["status", "--porcelain"], repo_dir):
            commit += "+dirty"
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit


def peak_rss_mb():
    """Peak resident memory of this process in MB, or None where it cannot be read."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if platform.system() == "Linux" else 1024      # Linux: kB; macOS, BSD: bytes
    return peak / scale / 1024


def setup_run_logging(outdir, level=logging.INFO):
    """Attaches log handlers to the ``"heracles"`` logger for one fit.

    ``<outdir>/heracles.log`` (appended to) receives every message at *level*; the
    stderr handler keeps any level set on the ``"heracles"`` logger beforehand
    (e.g. ``WARNING`` for a quiet console).  Returns ``(run_logger, handlers,
    prev_level)`` for :func:`teardown_run_logging`.
    """
    run_logger = logging.getLogger("heracles")
    prev_level = run_logger.level
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    file_h = logging.FileHandler(os.path.join(outdir, "heracles.log"),
                                 mode="a", encoding="utf-8")
    stream_h = logging.StreamHandler()
    # messages also raised as Python warnings go to the file only, not twice to the console
    stream_h.addFilter(lambda record: not getattr(record, "file_only", False))
    if prev_level != logging.NOTSET:
        stream_h.setLevel(prev_level)
    handlers = [file_h, stream_h]
    for h in handlers:
        h.setFormatter(fmt)
        run_logger.addHandler(h)
    run_logger.setLevel(level)
    return run_logger, handlers, prev_level


def teardown_run_logging(run_logger, handlers, prev_level):
    """Removes and closes the handlers added by :func:`setup_run_logging`."""
    for h in handlers:
        run_logger.removeHandler(h)
        h.close()
    run_logger.setLevel(prev_level)


def resolve_device(device):
    """Returns a torch.device for ``"cpu"``, ``"cuda"``, ``"cuda:N"``, a torch.device, or ``"auto"``.

    ``"auto"`` picks CUDA when available, else the CPU.
    """
    if device is None or device == "cpu":
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and (device == "cuda" or device.startswith("cuda:")):
        return torch.device(device)
    raise ValueError("device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got %r" % (device,))
