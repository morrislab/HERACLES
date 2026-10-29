"""Locate, build and run the vendored RapidNJ executable.

Resolution order for the binary (``rapidnj_executable``):

1. ``$HERACLES_RAPIDNJ`` (an explicit path);
2. ``heracles/_bin/rapidnj``, built when the package was installed;
3. a build in the user cache directory, compiled on first use from the
   vendored source in ``heracles/_vendor/rapidnj`` (needs a C++ compiler);
4. ``rapidnj`` on ``PATH``.

Without a binary, the initialisation falls back to MDS.  The vendored source
is patched to compile on non-x86 hosts (see ``_vendor/rapidnj/UPSTREAM.md``).
"""

import functools
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("heracles")


__all__ = [
    'VENDOR_DIR',
    'PACKAGED_BINARY',
    'SOURCES',
    'find_compiler',
    'source_hash',
    'build_rapidnj',
    'rapidnj_executable',
    'rapidnj_available',
    'run_rapidnj',
]


_PACKAGE_DIR = Path(__file__).resolve().parents[1]   # heracles/
VENDOR_DIR = _PACKAGE_DIR / "_vendor" / "rapidnj"
PACKAGED_BINARY = _PACKAGE_DIR / "_bin" / ("rapidnj.exe" if sys.platform == "win32" else "rapidnj")

# The upstream Makefile's object list (threadedNJ.cpp and sim_seq.cpp are not
# part of the executable).
SOURCES = (
    "main.cpp", "node.cpp", "distMatrixReader.cpp", "polytree.cpp", "diskMatrix.cpp",
    "rdDataInitialiser.cpp", "rapidNJ.cpp", "rapidNJDisk.cpp", "rapidNJMem.cpp",
    "simpleNJ.cpp", "testNJ.cpp", "ProgressBar.cpp", "getopt_pp/getopt_pp.cpp",
    "distanceCalculation/dnaBitString.cpp", "distanceCalculation/dataloaderPhylip.cpp",
    "distanceCalculation/hammingDistance.cpp", "distanceCalculation/bitDistanceGap.cpp",
    "distanceCalculation/JCdistance.cpp", "distanceCalculation/simpleDistanceCalculator.cpp",
    "distanceCalculation/dataloader.cpp", "distanceCalculation/KimuraDistance.cpp",
    "distanceCalculation/bitDistanceProtein.cpp", "distanceCalculation/DistanceEstimate.cpp",
)


def find_compiler():
    """Returns the path of a C++ compiler (``$CXX``, c++, g++, clang++), or None."""
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def source_hash():
    """Returns a short hash of the vendored sources, used to name the cache directory."""
    h = hashlib.sha1()
    for rel in sorted(p.relative_to(VENDOR_DIR).as_posix()
                      for p in (VENDOR_DIR / "src").rglob("*")
                      if p.suffix in (".cpp", ".h", ".hpp")):
        h.update(rel.encode())
        h.update((VENDOR_DIR / rel).read_bytes())
    return h.hexdigest()[:12]


def build_rapidnj(out_path, compiler=None, quiet=False):
    """Compiles the vendored RapidNJ into *out_path* and returns that path.

    Raises ``RuntimeError`` if no compiler is found, sources are missing, or the build fails.
    """
    compiler = compiler or find_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler found (tried $CXX, c++, g++, clang++)")
    src = VENDOR_DIR / "src"
    missing = [s for s in SOURCES if not (src / s).exists()]
    if missing:
        raise RuntimeError("vendored RapidNJ source incomplete: missing %s" % missing)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, "-O3", "-w", "-I%s" % src, "-I%s" % (src / "distanceCalculation"),
           "-pthread"] + [str(src / s) for s in SOURCES] + ["-o", str(out_path)]
    if not quiet:
        logger.info("building RapidNJ with %s -> %s", compiler, out_path)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("RapidNJ build failed:\n%s" % proc.stderr[-4000:])
    out_path.chmod(out_path.stat().st_mode | 0o111)
    return out_path


def _cache_binary():
    root = os.environ.get("HERACLES_CACHE") or os.environ.get("XDG_CACHE_HOME") \
        or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root) / "heracles" / ("rapidnj-" + source_hash()) / PACKAGED_BINARY.name


def _usable(path):
    return path is not None and os.path.isfile(path) and os.access(path, os.X_OK)


def rapidnj_executable(build=True):
    """Returns the path of a usable RapidNJ binary, in the order listed above.

    With ``build=True`` a missing binary is compiled into the user cache directory
    if a C++ compiler is present.  ``$HERACLES_RAPIDNJ`` is checked on every call;
    the rest of the lookup is cached per process.  ``FileNotFoundError`` if none.
    """
    explicit = os.environ.get("HERACLES_RAPIDNJ")
    if explicit:
        if _usable(explicit):
            return explicit
        raise FileNotFoundError("HERACLES_RAPIDNJ=%r is not an executable file" % explicit)
    return _resolve_binary(build)


def _resolve_binary(build):
    """Looks up the binary (packaged, cached, fresh build, PATH); raises if none."""
    path = _resolve_binary_cached(build)
    if path is None:
        raise FileNotFoundError(
            "no RapidNJ binary: set HERACLES_RAPIDNJ, put rapidnj on PATH, or install a "
            "C++ compiler so heracles can build its vendored copy (heracles falls back "
            "to an MDS initialisation without it).")
    return path


@functools.lru_cache(maxsize=None)
def _resolve_binary_cached(build):
    if _usable(PACKAGED_BINARY):
        return str(PACKAGED_BINARY)
    cached = _cache_binary()
    if _usable(cached):
        return str(cached)
    if build and VENDOR_DIR.is_dir() and find_compiler():
        try:
            build_rapidnj(cached)
            return str(cached)
        except RuntimeError as e:
            logger.warning("could not build the vendored RapidNJ: %s", e)
    on_path = shutil.which("rapidnj")
    if _usable(on_path):
        return on_path
    return None


def rapidnj_available():
    """Returns True if a usable RapidNJ binary is found; never raises."""
    try:
        return rapidnj_executable() is not None
    except FileNotFoundError:
        return False


def run_rapidnj(phylip_path, *, timeout=None):
    """Runs RapidNJ on a PHYLIP distance-matrix file; returns the CompletedProcess.

    *timeout* is in seconds (default: none).  Raises ``RuntimeError``, with
    RapidNJ's stderr, if it exits with an error or times out.
    """
    cmd = [rapidnj_executable(), str(phylip_path), "-i", "pd", "-o", "t"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True,
                              timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("RapidNJ failed (exit %d) on %s:\n%s"
                           % (e.returncode, phylip_path, (e.stderr or "")[-2000:])) from e
    except subprocess.TimeoutExpired as e:
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        raise RuntimeError("RapidNJ exceeded %s s on %s:\n%s"
                           % (timeout, phylip_path, stderr[-2000:])) from e
