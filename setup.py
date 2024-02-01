from setuptools import setup
from version import get_git_version

setup(name='lwa-fasttransients',
      version=get_git_version(),
      url='https://github.com/ovro-lwa/lwa-fasttransients',
      python_requires='~=3.9.18',
      zip_safe=False)