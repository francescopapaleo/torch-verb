from setuptools import setup, find_packages

setup(
    name='torchverb',
    version='0.0.1',
    author='Francesco Papaleo',
    author_email='francesco.papaleo@gmail.com',
    description='A package for reverb algorithms in PyTorch',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'matplotlib',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Affero GNU General Public License v3 (AGPL-3.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
