# -*- coding: utf-8 -*-

"""
MINOAN: DDPSE Lab @ GT 
Copyright (C) 2020 - Sophie Kim 

Test problem ex1221 
Retrieved from: http://www.minlplib.org/ex1221.html
Gray-box problem: con1 and con5 are known 
"""

from minoan import MINOAN
import numpy as np
import pyomo.environ as pe 


def simulator(x):

    obj = - (- 2*x[0] - 3*x[1] - 1.5*x[2] - 2*x[3] + 0.5*x[4])
    # con1 = x[0]**2 + x[2] 
    con2 = x[1]**1.5 + 1.5 * x[3] 
    con3 = x[0] + x[2] 
    con4 = -(1.333*x[1] + x[3])
    # con5 = -x[2] - x[3] + x[4] 
    return np.array([obj, con2, con3, con4])

# define gray box constraints; must be in this format 
def graycons(model):
    # add known constraints 
    model.gray = pe.ConstraintList()
    model.gray.add(model.var[0]**2 + model.var[2] == 1.25)
    model.gray.add(-model.var[2] - model.var[3] + model.var[4] <= 0)
    return model 


vartype = ['R','R','B','B','B']
lb = [0,0,0,0,0]
ub = [10,10, 1, 1, 1]
modeltype = 'hybrid'
onehotencoding = 1
nprocs = 1
maxeval = 3000
contype = ['E','L','G']
conrhs = [3, 1.6, -3]

solver = 'neos' # you can change this to gams if you have access to gams 

opt = MINOAN(vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, solver, put = 0)
stat, xbest, ybest, vio = opt.main()

