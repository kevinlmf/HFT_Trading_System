# setup.py - Build configuration for top-level module: cpp_trading2
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys, os, glob

# Try to get pybind11 include directories
def get_pybind_includes():
    try:
        import pybind11
        return [pybind11.get_include(), pybind11.get_include(user=True)]
    except Exception:
        return []

# Source files
sources = [
    "cpp_core/bindings/all_bindings.cpp",
    *glob.glob("cpp_core/src/*.cpp"),  # data_feed.cpp, order.cpp, order_executor.cpp
]

include_dirs = [
    "cpp_core/include",
    "cpp_core",  # In case headers use relative includes
    *get_pybind_includes(),
]

extra_compile_args = ["-std=c++17", "-O3"]
if sys.platform == "darwin":
    # macOS supports both ARM/Intel; can add minimum version if needed:
    # extra_compile_args += ["-mmacosx-version-min=11.0"]
    pass
elif sys.platform.startswith("linux"):
    pass
elif os.name == "nt":
    # Windows MSVC
    extra_compile_args = ["/std:c++17", "/O2"]

ext_modules = [
    Extension(
        name="cpp_trading2",              # Top-level module name: cpp_trading2
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="cpp_trading2",
    version="0.1.0",
    description="C++ trading core (pybind11) for quantitive-trading-system",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
