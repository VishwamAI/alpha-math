from setuptools import setup, find_packages

setup(
    name="alpha-math",
    version="0.1.0",
    author="VishwamAI",
    author_email="your.email@example.com",
    description="A library for advanced mathematical computations and reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/alpha-math",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.3",
        "torch>=2.4.0",
        "gym>=0.18.3",
        "matplotlib>=3.4.2",
        "mathematics-dataset==1.0.1",  # Pin to exact version for compatibility
        "sympy==1.6.2",  # Pin to exact version that has base_solution_linear
        "six>=1.16.0",  # Required by mathematics-dataset
        "absl-py>=0.1.0",  # Required by mathematics-dataset
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
)
