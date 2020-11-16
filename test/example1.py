# -*- coding: utf-8 -*-

'''
MINOAN: DDPSE Lab @ GT 
Copyright (C) 2020 - Sophie Kim 

Test problem ex1221 
Retrieved from: http://www.minlplib.org/ex1221.html
Black-box problem: all constraints and objective unknown 
'''

from minoan import MINOAN
import numpy as np


def simulator(x):
    # black box simulation 
    obj = - (- 2*x[0] - 3*x[1] - 1.5*x[2] - 2*x[3] + 0.5*x[4])
    con1 = x[0]**2 + x[2] 
    con2 = x[1]**1.5 + 1.5 * x[3] 
    con3 = x[0] + x[2] 
    con4 = -(1.333*x[1] + x[3])
    con5 = -x[2] - x[3] + x[4] 
    return np.array([obj, con1, con2, con3, con4, con5])

graycons = None

# variable type - R for continuous and B for binary
# real (continuous) variables should be specified before binary variables
# for example, ['R','B','R'] is not acceptable. It should be ['R','R','B']
vartype = ['R','R','B','B','B']

# lower and upper bounds
lb = [0,0,0,0,0]
ub = [10,10, 1, 1, 1]

# ML model type; options = hybrid, ANN, GP, SVR
modeltype = 'ANN'
onehotencoding = 1
nprocs = 1

# max number of simulation evaluation allowed
maxeval = 3000

# constraint information; E for equality constraint; G for >=; L for <= 
contype = ['E','E','L','G','L']
conrhs = [1.25, 3, 1.6, -3, 0]

solver = 'neos' # you can change this to gams if you have access to gams 
# optimize! 
opt = MINOAN(vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, solver, put = 0)
stat, xbest, ybest, vio = opt.main()
