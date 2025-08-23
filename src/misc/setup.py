from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="computational_physics",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A collection of computational physics experiments and utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/computational_physics_experiments",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "numba>=0.53.0",
        "cython>=0.29.24",
        "sympy>=1.8.0",
        "scikit-image>=0.18.0",
        "pytest>=6.2.5"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    keywords="physics computational-physics scientific-computing simulation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/computational_physics_experiments/issues",
        "Source": "https://github.com/yourusername/computational_physics_experiments",
    },
)
