from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pillow']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)