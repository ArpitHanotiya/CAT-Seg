from setuptools import setup, find_packages

setup(
    name="catseg",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "clip",
        "opencv-python",
        "timm",
    ],
)