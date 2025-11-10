from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cpp_trading2",  # 和setup中的name保持一致
        [
            "bindings/all_bindings.cpp",   # ✅ Merge绑定
            "src/data_feed.cpp",
            "src/order.cpp",
            "src/slippage_calculator.cpp"  # Slippage计算器实现
        ],
        include_dirs=["include"],
    ),
]

setup(
    name="cpp_trading2",
    version="0.2",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.10"],
    setup_requires=["pybind11>=2.10"],
)


