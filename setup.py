from setuptools import find_packages
from setuptools import setup

setup(name='TPS',
      install_requires=['pyyaml', 'tensorboardX',
                        'easydict', 'matplotlib',
                        'scipy', 'scikit-image',
                        'future', 'setuptools',
                        'tqdm', 'cffi'],
      packages=find_packages())
