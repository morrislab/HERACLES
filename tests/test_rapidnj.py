"""Tests for locating, building and running the vendored RapidNJ binary."""

import stat

import pytest


def test_rapidnj_executable_prefers_explicit_override(monkeypatch, tmp_path):
    from heracles.trees.rapidnj import rapidnj_executable

    executable = tmp_path / "rapidnj"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERACLES_RAPIDNJ", str(executable))

    assert rapidnj_executable() == str(executable)


def test_rapidnj_clamped_is_the_public_default():
    from heracles.config import HeraclesConfig

    assert HeraclesConfig().init_method == "rapidnj_clamped"


def test_rapidnj_reports_how_to_get_a_binary_when_nothing_is_available(
        monkeypatch, tmp_path):
    import heracles.trees.rapidnj as rapidnj

    monkeypatch.delenv("HERACLES_RAPIDNJ", raising=False)
    monkeypatch.setattr(rapidnj, "PACKAGED_BINARY", tmp_path / "missing")
    monkeypatch.setattr(rapidnj, "_cache_binary", lambda: tmp_path / "missing-cache")
    monkeypatch.setattr(rapidnj, "find_compiler", lambda: None)
    monkeypatch.setattr(rapidnj.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="HERACLES_RAPIDNJ"):
        rapidnj.rapidnj_executable()


def test_vendored_source_ships_inside_the_package():
    from heracles.trees.rapidnj import SOURCES, VENDOR_DIR

    assert VENDOR_DIR.parts[-3:] == ("heracles", "_vendor", "rapidnj")
    assert all((VENDOR_DIR / "src" / s).exists() for s in SOURCES)


@pytest.mark.slow
def test_build_rapidnj_produces_a_working_binary(tmp_path):
    from heracles.trees.rapidnj import build_rapidnj, find_compiler
    import subprocess

    if find_compiler() is None:
        pytest.skip("no C++ compiler")
    exe = build_rapidnj(tmp_path / "rapidnj", quiet=True)
    phy = tmp_path / "t.phy"
    phy.write_text("4\nA 0 2 4 6\nB 2 0 4 6\nC 4 4 0 6\nD 6 6 6 0\n")
    out = subprocess.run([str(exe), str(phy), "-i", "pd", "-o", "t"],
                         capture_output=True, text=True, check=True).stdout
    assert "A" in out and out.strip().endswith(";")


def test_run_rapidnj_has_no_default_timeout():
    import inspect
    from heracles.trees.rapidnj import run_rapidnj

    assert inspect.signature(run_rapidnj).parameters["timeout"].default is None


def test_run_rapidnj_reports_stderr_on_failure(monkeypatch, tmp_path):
    from heracles.trees.rapidnj import run_rapidnj

    fake = tmp_path / "rapidnj"
    fake.write_text("#!/bin/sh\necho 'bad matrix' >&2\nexit 3\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERACLES_RAPIDNJ", str(fake))
    with pytest.raises(RuntimeError, match=r"(?s)exit 3.*bad matrix") as e:
        run_rapidnj(tmp_path / "in.phy")
    assert "in.phy" in str(e.value)


def test_rapidnj_lookup_is_cached_but_env_override_wins(monkeypatch, tmp_path):
    import heracles.trees.rapidnj as rapidnj

    binary = tmp_path / "rapidnj"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.delenv("HERACLES_RAPIDNJ", raising=False)
    monkeypatch.setattr(rapidnj, "PACKAGED_BINARY", binary)
    calls = []
    real = rapidnj._cache_binary
    monkeypatch.setattr(rapidnj, "_cache_binary", lambda: calls.append(1) or real())
    assert rapidnj.rapidnj_executable() == str(binary)
    assert rapidnj.rapidnj_executable() == str(binary)
    assert calls == []                      # packaged binary found, cache never probed

    monkeypatch.setattr(rapidnj, "PACKAGED_BINARY", tmp_path / "missing")
    assert rapidnj.rapidnj_executable() == str(binary)   # cached answer survives the patch

    other = tmp_path / "other"
    other.write_text("#!/bin/sh\nexit 0\n")
    other.chmod(other.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERACLES_RAPIDNJ", str(other))
    assert rapidnj.rapidnj_executable() == str(other)
