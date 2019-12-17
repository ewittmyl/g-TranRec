from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "g-TranRec",
    version = "0.1.1",
    author = "Y.-L.Mong",
    author_email = "yik.mong@monash.edu",
    description = "GOTO transient searching",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
