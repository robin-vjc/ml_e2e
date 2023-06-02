from setuptools import find_packages, setup

with open("requirements/base.txt", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="ml_e2e",
    version="1.0",
    description="Machine Learning Project - End to End Blueprint",
    author="Robin Vujanic",
    author_email="vjc.robin@gmail.com",
    packages=find_packages(),
    install_requires=required,
)
