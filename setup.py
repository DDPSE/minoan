# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:05:37 2020

@author: k3148
"""


from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sbopt',
    url='https://github.gatech.edu/boukouvala-lab/sbopt',
    author='Sophie Kim',
    author_email='skim3061@gatech.edu',
    # Needed to actually package something
    packages=['sbopt'],
    # Needed for dependencies
    # install_requires=['numpy','pandas','pyomo'],
    # *strongly* suggested for sharing
    version='0.0.1',
    # The license can be anything you like
    license='GT',
    description='surrogate-based optimization',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)