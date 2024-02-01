from setuptools import setup, find_packages

setup(
    name="drift-lens",
    version="0.1.1",
    packages=find_packages(),
    description="DriftLens: an Unsupervised Drift Detection framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Salvatore Greco",
    author_email="grecosalvatore94@gmail.com",
    url="https://github.com/grecosalvatore/drift-lens",
    install_requires=[
        # List your package dependencies here
         'numpy', 'pandas',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)