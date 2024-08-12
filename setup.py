from setuptools import setup, find_packages

setup(
    name="driftlens",
    version="0.1.5",
    packages=find_packages(),
    description="DriftLens: an Unsupervised Drift Detection framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Salvatore Greco",
    author_email="grecosalvatore94@gmail.com",
    url="https://github.com/grecosalvatore/drift-lens",
    install_requires=[
        "numpy>=1.22.4",
        "scikit-learn>=0.24.2",
        "matplotlib~=3.5.1",
        "pandas>=1.1.3",
        "scipy>=1.5.0",
        "tqdm~=4.64.1",
        "setuptools~=58.0.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)