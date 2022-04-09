from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nsdmd",
    version="0.0.1",
    author="John Ferre",
    author_email="jbferre@uw.edu",
    description="Non Stationary Dynamical Mode Decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jbferre/nsdmd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.8',
    install_requires=["numpy","colorednoise"]
)