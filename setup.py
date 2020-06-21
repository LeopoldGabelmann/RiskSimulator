from setuptools import setup, find_packages
import settings


def readme():
    with open("README.md") as f:
        return f.read()


with open("requirements.txt") as f:
    base_packages = f.read().splitlines()


with open("requirements-dev.txt") as f:
    dev_packages = [p for p in f.read().splitlines() if "requirements.txt" not in p]
dev_packages = base_packages + dev_packages


setup(
    name="risksimulator",
    version=settings.VERSION_NUMBER,
    description=settings.PROJECT_DESCRIPTION,
    url=settings.PROJECT_URL,
    author=settings.PROJECT_AUTHORS,
    author_email=settings.PROJECT_AUTHORS_EMAIL,
    packages=find_packages(),
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
    python_requires="3.8.1",
    test_suite="pytest-runner",
    tests_require=["pytest"],
    include_package_data=True,
)
