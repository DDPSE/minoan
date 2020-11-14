# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:05:37 2020

@author: k3148
"""


from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sbopt',
    version='0.0.1',
    license = 'GT',
    description='Mixed INteger Optimization using ApproximatioNs',
    url='https://github.com/DDPSE/minoan',
    author='Sophie Kim, Fani Boukouvala',
    author_email='sophiekim0205@gmail.com, fani.boukouvala@chbe.gatech.edu',
    packages=['minoan'],
    install_requires=['numpy','pandas','pyomo','pyDOE','scipy','scikit-learn','joblib'],

)