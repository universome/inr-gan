#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/rakhimovv/deformable_inr'
install_requires = [
    'hydra-core',
    'pytorch-lightning'
]

setup(name='dinr',
      version=__version__,
      description='deformable inr',
      author='3ddl',
      author_email='',
      url=url,
      install_requires=install_requires,
      packages=find_packages()
      )
