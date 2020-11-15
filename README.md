# MINOAN
Mixed INteger Optimization using ApproximatioNs
(Beta Version 0.0.1)

## About 
MINOAN is an open-source Python library used for machine learning-based (or surrogate-based) optimization. The alglorithm supports constrained *NLP* and *MINLP* (with binary variables) problems. It currently supports the following machine learning models:
* Artificial Neural Network (tanh and relu activation function)
* Gaussian Process
* Support Vector Regression 

These models are constructed using scikit-learn and optimized using Pyomo via GAMS or NEOS interface. MINOAN has additional capabilities such as: 
* Parallel processing for multiple promising binary solutions 
* Gray-box problems with known/explicit constraints

If you have any questions or concerns, please send an email to sophiekim0205@gmail.com or fani.boukouvala@chbe.gatech.edu

## Installation

If using Anaconda, first run: 
conda install git pip

The code can be directly installed from github using the following command: 
pip install git+git://github.com/DDPSE/minoan

## Examples 
Example codes are found in the directory "test". 
* Example 1: constrained, black-box MINLP problem 
* Example 2: constrained, gray-box MINLP problem
* Example 3: constrained, black-box NLP problem

## References
* Kim SH, Boukouvala F. Machine learning-based surrogate modeling for data-driven optimization: a comparison of subset selection for regression techniques. Optimization Letters. 2019.
* Kim SH, Boukouvala F. Surrogate-Based Optimization for Mixed-Integer Nonlinear Problems. Computers & Chemical ENgineering. 2020. 
