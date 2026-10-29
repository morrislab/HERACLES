"""Build the vendored RapidNJ binary into the package at install time.

Everything else is declared in pyproject.toml.  A missing compiler is not an
error: the package still installs and builds RapidNJ on first use (or falls
back to an MDS initialisation).
"""

import importlib.util
import os
import warnings

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as _build_py

HERE = os.path.dirname(os.path.abspath(__file__))


def _rapidnj_module():
    spec = importlib.util.spec_from_file_location(
        "heracles_rapidnj_build", os.path.join(HERE, "heracles", "trees", "rapidnj.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class build_py(_build_py):
    def run(self):
        super().run()
        rapidnj = _rapidnj_module()
        target = os.path.join(self.build_lib, "heracles", "_bin", rapidnj.PACKAGED_BINARY.name)
        try:
            rapidnj.build_rapidnj(target, quiet=True)
            print("built RapidNJ -> %s" % target)
        except RuntimeError as e:
            warnings.warn("RapidNJ was not built at install time (%s); heracles will "
                          "build it on first use or fall back to MDS initialisation" % e)


class BinaryDistribution(Distribution):
    """The wheel carries a compiled RapidNJ, so tag it for this platform."""

    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": build_py}, distclass=BinaryDistribution)
