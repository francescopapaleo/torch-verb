from pathlib import Path
from setuptools import setup, find_packages


def get_readme_text():
    root_dir = Path(__file__).parent
    readme_path = root_dir / "README.md"
    return readme_path.read_text()


setup(
    name="torchverb",
    version="0.0.1",
    description="A package for reverb algorithms in PyTorch",
    long_description=get_readme_text(),
    long_description_content_type="text/markdown",
    author="Francesco Papaleo",
    url="",
    license="AGPLv3",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "matplotlib",
        "pytest",
    ],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Affero GNU General Public License v3 (AGPL-3.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
