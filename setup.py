from setuptools import setup, find_packages

setup(
    name="lvec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "awkward>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    author="Mohamed Elashri",
    author_email="lvec@elashri.com",
    description="A package for handling Lorentz vectors with NumPy and Awkward array backends",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    keywords="physics,lorentz,vector,numpy,awkward",
    url="https://github.com/yourusername/lvec",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)