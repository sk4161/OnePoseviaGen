# src/fpose/setup.py
from setuptools import setup, find_packages

setup(
    name='amodal3r',
    version='0.1',
    package_dir={'amodal3r': 'amodal3r'},                  
    packages=find_packages(where='amodal3r', include=['amodal3r*']),
)