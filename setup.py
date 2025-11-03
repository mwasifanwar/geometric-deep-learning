from setuptools import setup, find_packages

setup(
    name="geometric-deep-learning-3d",
    version="1.0.0",
    author="mwasifanwar",
    description="Advanced 3D scene reconstruction and understanding using geometric deep learning on point clouds and meshes",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "open3d>=0.17.0",
        "scipy>=1.10.0",
        "pytest>=7.3.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0"
    ],
    python_requires=">=3.8",
)