# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:01:11 2020

@author: k3148
"""
#ex1221

import sys
from utils import info
from main import *
import numpy as np
import pyomo.environ as pe 

# set algorithm path 
# AlgorithmPath = '/nv/hp13/skim3061/data/BBOPT/ALGORITHMS/final'
AlgorithmPath= r'C:\Users\k3148\Dropbox (GaTech)\Sophie_Paper4\Final code'
# AlgorithmPath = r'W:\data\BBOPT\ALGORITHMS\final'
sys.path.append(AlgorithmPath)


def simulator(x):
    # takes in a np array of input data 
    # returns np array of output (obj, con...)
    obj = - (- 2*x[0] - 3*x[1] - 1.5*x[2] - 2*x[3] + 0.5*x[4])
    con1 = x[0]**2 + x[2] 
    con2 = x[1]**1.5 + 1.5 * x[3] 
    con3 = x[0] + x[2] 
    con4 = -(1.333*x[1] + x[3])
    con5 = -x[2] - x[3] + x[4] 
    return np.array([obj, con1, con2, con3, con4, con5])
    # return np.array([obj, con2, con3, con4])

# # import pyomo.environ as pe 
# # def graycons(model):
# #     # add known constraints 
# #     model.gray = pe.ConstraintList()
# #     model.gray.add(model.var[0]**2 + model.var[2] == 1.25)
# #     model.gray.add(-model.var[2] - model.var[3] + model.var[4] <= 0)
# #     return model 

graycons = None

vartype = ['R','R','B','B','B']
lb = [0,0,0,0,0]
ub = [10,10, 1, 1, 1]
modeltype = 'hybrid'
onehotencoding = 1
nprocs = 1
maxeval = 3000
contype = ['E','E','L','G','L']
conrhs = [1.25, 3, 1.6, -3, 0]

# contype = ['E','L','G']
# conrhs = [3, 1.6, -3]


opt = minoan(vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, 'gams', put = 1)
stat, xbest, ybest, vio = opt.main()




# def simulator(x):
#     # takes in a np array of input data 
#     # returns np array of output (obj, con...)
#     x1 = x[0]
#     x2 = x[1]
#     x3 = x[2]
#     x4 = x[3]
#     x5 = x[4]
#     x6 = x[5]
#     obj = -(-(6.5*x1 - 0.5*x1*x1) + x2 + 2*x3 + 3*x4 + 2*x5 + x6)
#     con1 = x1 + 2*x2 + 8*x3 + x4 + 3*x5 + 5*x6
#     con2 = - 8*x1 - 4*x2 - 2*x3 + 2*x4 + 4*x5 - x6
#     con3 =  2*x1 + 0.5*x2 + 0.2*x3 - 3*x4 - x5 - 4*x6
#     con4 = 0.2*x1 + 2*x2 + 0.1*x3 - 4*x4 + 2*x5 + 2*x6 
#     con5 = - 0.1*x1 - 0.5*x2 + 2*x3 + 5*x4 - 5*x5 + 3*x6
#     return np.array([obj, con1, con2, con3, con4, con5])
#     # return np.array([obj, con2, con3, con4])

# # import pyomo.environ as pe 
# # def graycons(model):
# #     # add known constraints 
# #     model.gray = pe.ConstraintList()
# #     model.gray.add(model.var[0]**2 + model.var[2] == 1.25)
# #     model.gray.add(-model.var[2] - model.var[3] + model.var[4] <= 0)
# #     return model 

# graycons = None

# vartype = ['R','R','R','R','R','R']
# lb = [0,0,0,0,0,0]
# ub = [1, 7, 1, 1, 1, 2]
# modeltype = 'ANN'
# onehotencoding = 0
# nprocs = 1
# maxeval = 3000
# contype = ['L','L','L','L','L']
# conrhs = [16,-1,24,12,3]
