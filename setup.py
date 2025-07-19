#!/usr/bin/env python

from setuptools import setup

setup(
    name='smude',
    version='0.1.0',
    description='Sheet Music Dewarping',
    author='Simon Waloschek',
    url='https://github.com/sonovice/smude',
    packages=['smude'],

    entry_points={
        'console_scripts': [
            'smude = smude:main',
        ],
    },

    python_requires='>=3.12',

    install_requires=[
        'numpy>=1.25',
        'torch>=2.3.0',
        'pytorch-lightning>=2.2.0',
        'scikit-image>=0.22',
        'scipy>=1.11',
        'torchvision>=0.18',
        'typing_extensions>=4.10',
        'tqdm>=4.48',
        'requests>=2.31',
        'opencv-contrib-python>=4.9'
    ],
)
