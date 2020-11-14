# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:38:54 2020

@author: k3148
"""

'''
NLP Test problem 
Retrieved from: http://www.minlplib.org/
'''

from minoan import MINOAN
import numpy as np

def simulator(x):

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    obj = -(-(6.5*x1 - 0.5*x1*x1) + x2 + 2*x3 + 3*x4 + 2*x5 + x6)
    con1 = x1 + 2*x2 + 8*x3 + x4 + 3*x5 + 5*x6
    con2 = - 8*x1 - 4*x2 - 2*x3 + 2*x4 + 4*x5 - x6
    con3 =  2*x1 + 0.5*x2 + 0.2*x3 - 3*x4 - x5 - 4*x6
    con4 = 0.2*x1 + 2*x2 + 0.1*x3 - 4*x4 + 2*x5 + 2*x6 
    con5 = - 0.1*x1 - 0.5*x2 + 2*x3 + 5*x4 - 5*x5 + 3*x6
    return np.array([obj, con1, con2, con3, con4, con5])
 

graycons = None

vartype = ['R','R','R','R','R','R']
lb = [0,0,0,0,0,0]
ub = [1, 7, 1, 1, 1, 2]
modeltype = 'ANN'
onehotencoding = 0
nprocs = 1
maxeval = 3000
contype = ['L','L','L','L','L']
conrhs = [16,-1,24,12,3]
solver = 'gams' # or neos if gams is not available 

opt = MINOAN(vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, solver, put = 0)
stat, xbest, ybest, vio = opt.main()
