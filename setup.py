from setuptools import setup, Extension

module = Extension("_world_logic", sources=["_world_logic.c"])

setup(
    name="_world_logic",
    version="1.0",
    ext_modules=[module],
)