# Builds src/hptp.cpp into hptpermute/_hptp (a shared library loaded with ctypes). Metadata is in pyproject.toml.
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            args = ["/std:c++17", "/O2", "/EHsc"]
        else:
            args = ["-std=c++17", "-O2", "-pthread", "-fvisibility=hidden"]
        for ext in self.extensions:
            ext.extra_compile_args = args
            if self.compiler.compiler_type != "msvc" and sys.platform != "darwin":
                ext.extra_link_args = ["-pthread"]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "hptpermute._hptp",
            sources=["src/hptp.cpp"],
            include_dirs=["include"],
            define_macros=[("HPTP_SHARED", None), ("HPTP_BUILDING", None), ("HPTP_PYTHON_EXTENSION", None)],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
