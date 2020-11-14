# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:29:04 2020

@author: k3148
"""


import numpy as np
import pandas as pd
import os
import time 
import subprocess 
from scipy.spatial import distance 
import random
import pyomo.environ as pe

class pyomoopt:
    '''
    Main class used for surrogate model optimization. The optimization model is
    created by using pyomo. Different solver interface can be used to optimize
    the model (GAMS or NEOS suggested).
    
    Args: 
        prob: class that contains problem information
        BestModel: Best sklearn model; generated from modelopt class 
    ''' 
    
    def __init__(self, prob, BestModel):
        self.prob = prob
        self.BestModel = BestModel
        self.onehotencoding = prob.onehotencoding # mixed-integer model 
        self.solver = self.prob.solver # solver interface
    
    ''' Define parameters for selected ML model type ''' 
    def ANNParameters(self):
        # for ANN model 
        self.hidden_w_all = self.BestModel['Model'].coefs_ # NN weights
        self.hidden_b_all = self.BestModel['Model'].intercepts_ # NN bias
        self.actfunc = self.BestModel['Model'].activation # activation function
        self.InputDimension = self.hidden_w_all[0].shape[0] # num of input nodes
        self.OutputDimension = self.hidden_w_all[1].shape[1] # num of output nodes
        
    def GPParameters(self):
        # for GP models
        self.coef = self.BestModel['Model'].alpha_ #coefficient 
         #training data set; used to construct cov matrix
        self.xtrain = self.BestModel['Model'].X_train_
        self.InputDimension = self.xtrain.shape[1] # num of input 
        self.OutputDimension = self.coef.shape[1] # num of output 
        
    def SVRParameters(self):
        # for SVR models 
        self.gamma = [] 
        self.b = []
        self.sv = {} 
        self.coef = {} 
        for i in self.BestModel['Model']:
            model = self.BestModel['Model'][i]
            
            self.gamma.append(model.gamma) #gamma
            self.b.append(model.intercept_) #bias
            self.sv[i] = model.support_vectors_ #support vectors
            self.coef[i] = model.dual_coef_.flatten() #coefficients
            
        self.InputDimension = self.sv[0].shape[1] #num of input 
        self.OutputDimension = len(self.BestModel['Model']) # num of output
    
    ''' General functions for defining variable type in pyomo format''' 
    def DefineVariableType(self):
        VariableType0 = self.prob.vartype.copy()
        # for one hot encoding: add additional binary variables for dummy
        if self.onehotencoding:
            for ind, val in enumerate(self.prob.vartype):
                if val == 'B':
                    VariableType0.insert(ind, 'B')
        # specify variable type in pyomo form
        VariableType = {} 
        for ind, val in enumerate(VariableType0):
            if val == 'R':
                VariableType[ind] = pe.Reals
            elif val == 'B':
                VariableType[ind] = pe.Binary
    
        return VariableType
    
    
    def DefineVariable(self, model):
        '''
        Define pyomo variable type 
        
        Args:
            model: pyomo model 
        Returns:
            model: pyomo model with input variables specified
        ''' 
        # call necessary parameters
        if self.prob.modeltype == 'ANN':
            self.ANNParameters()
        elif self.prob.modeltype == 'GP':
            self.GPParameters()
        elif self.prob.modeltype == 'SVR':
            self.SVRParameters()
            
        # define variable type 
        VariableType = self.DefineVariableType()

        # determine input dimension; store as model.ind
        if len(self.prob.binsol) > 0: 
            model.ind = range(self.InputDimension + len(self.prob.binsol))
        else: 
            model.ind = range(self.InputDimension)
        
        # function for setting variable type; required for pyomo 
        def SetVariableType(model, i):
            return VariableType[i]
        
        # define model.x = input variable 
        model.x = pe.Var(model.ind, within = SetVariableType, bounds = (0,1))
        return model
    
    def scaleconrhs(self, rhs, scaler):
        '''
        Scale rhs constraint value for optimization 
        
        Args: 
            rhs: constraint rhs 
            scaler: scaler class; used to access data min and max 
            
        Returns:
            scaledRHS: scaled rhs constraint 
        ''' 
        scaledRHS = [(rhs[i]-scaler.data_min_[i+1])/(scaler.data_max_[i+1] - scaler.data_min_[i+1]) \
                     for i in range(len(self.prob.conrhs))]
        return scaledRHS
    

    
    ''' 
    ===================================================
    Functions used for ANN model construction in pyomo 
    ===================================================
    ''' 
    def hiddenlayers(self, model, w, b, actfunc):
        ''' 
        Construct a neural network
        For input -> hidden layer
        
        Args: 
            model: pyomo model
            w: ANN weights
            b: ANN bias
            actfunc: activation function 
        Returns:
            model: pyomo model that contains ANN 
        '''
        
        # Construct ANN equation using weights, bias, and activation function  
        
        # hyperbolic tangent activation function
        def pe_tanh(x):
            return 1 - 2/(pe.exp(2*x) + 1)
        
        # sigmoid act function
        def pe_sigmoid(x):
            return 1/(1 + pe.exp(-x))
        
        # relu activation function 
        def pe_relu(x):
            # if relu, just return x and later reformulate into linear constraints
            return x
        
        # set activation function 
        if actfunc == 'tanh':
            pe_actfunc = pe_tanh
        elif actfunc == 'sigmoid':
            pe_actfunc = pe_sigmoid
        elif actfunc == 'relu':
            pe_actfunc = pe_relu
        
        # # of input and hidden nodes         
        ninputnode = w.shape[0]
        nhiddennode = w.shape[1]
        model.hind = range(nhiddennode)
        
        model.a = pe.Var(model.hind, within = pe.Reals, initialize = 1)
        model.acon = pe.ConstraintList()

        model.h = pe.Var(model.hind, within = pe.Reals, initialize = 1)
        
        if pe_actfunc == 'relu':
            # for relu; reformulate into 5 linear constraints 
            model.h1 = pe.ConstraintList()
            model.h2 = pe.ConstraintList()
            model.h3 = pe.ConstraintList()
            model.h4 = pe.ConstraintList()
            model.h5 = pe.ConstraintList()
        else: # for sigmoid and tanh 
            model.hcon = pe.ConstraintList()

        model.yh = pe.Var(model.hind, within = pe.Binary)
        
        for j in range(nhiddennode):
            model.acon.add((sum([model.x[i] * w[i,j] for i in range(ninputnode)])+ b[j]) == model.a[j])

            if pe_actfunc == 'relu':
                M = 20 # M constraint - arbitrarily large number 
                
                model.h1.add(model.h[j] >= 0) 
                model.h2.add(model.h[j] <= M)
                model.h3.add(model.h[j] >= model.a[j])
                model.h4.add(model.h[j] <= M * model.yh[j])
                model.h5.add(model.h[j] <= model.a[j] + M * (1-model.yh[j]))
        
            else: # for sigmoid and tanh
                model.hcon.add(pe_actfunc(model.a[j]) == model.h[j])
        
        return model 
    
    
    def outputlayer(self, model, w, b, sl):
        ''' 
        Construct a neural network
        For hidden -> output 
        
        Args: 
            model: pyomo model
            w: ANN weights
            b: ANN bias
            sl: list of slack variables; used if infeasibility problem has been solved
        Returns:
            model: pyomo model that contains ANN 
        '''
        
        # number of hidden and output nodes
        nhiddennode = w.shape[0]
        noutputnode = w.shape[1]
        
        model.outind = range(noutputnode)
        model.out = pe.Var(model.outind, within = pe.Reals, initialize = 1)
        
        model.outcon = pe.ConstraintList()
        # scale constraint rhs 
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)
        
        for k in range(noutputnode):
            if k == 0: # for objective     
                model.obj = pe.Objective(expr = sum([model.h[j] * w[j,k] for j in range(nhiddennode)]) + b[k], sense = pe.minimize)
            else:
                ConType = self.prob.contype[k-1]
                f = sum([model.h[j] * w[j,k] for j in range(nhiddennode)]) + b[k] + sl[k-1]
                
                if ConType == 'E':
                    # reformulate equality constraint into two inequality constraints
                    model.outcon.add(f <= conrhs[k-1])
                    model.outcon.add(f >= conrhs[k-1])
                
                elif ConType == 'L': #<= constraint
                     model.outcon.add(f <= conrhs[k-1])
                
                elif ConType == 'G': #>= constraint
                     model.outcon.add(f >= conrhs[k-1])
        
        return model
    
    def outputlayer_infeas(self, model, w, b):
        
        ''' 
        ANN infeasibility problem 
        Add slack variable to find the most feasible solution
        
        Args:
            model: pyomo model
            w: weights
            b: bias
        Returns:
            model: pyomo model 
        ''' 
        
        nhiddennode = w.shape[0]
        noutputnode = w.shape[1] 
        
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)

        model.outind = range(noutputnode)
        model.out = pe.Var(model.outind, within = pe.Reals, initialize = 1)
        model.sl = pe.Var(model.outind, within = pe.Reals, bounds = (0,0.1))
        
        model.outcon = pe.ConstraintList()
         
        # for all constraints
        for k in range(noutputnode):
            ConType = self.prob.contype[k]
            
            f = sum([model.h[j] * w[j,k] for j in range(nhiddennode)]) + b[k] - model.sl[k]
            if ConType == 'E':
                model.outcon.add(f <= conrhs[k])
                model.outcon.add(f >= conrhs[k])
            elif ConType == 'L':
                model.outcon.add(f <= conrhs[k])
            elif ConType == 'G':
                model.outcon.add(f >= conrhs[k]) 
        
        # new objective -- minimize the sum of slack variables 
        model.obj = pe.Objective(expr = sum([model.sl[k] for k in range(noutputnode)]))
    
        return model
    
    def optimizeANN(self, multistartloc, slack = None, optall = 1):
        
        ''' 
        Main function for ANN optimization 
        
        Args:
            multistartloc: x sample used for initialization for multistart opt 
            slack: slack variable; default = None 
            optall: perform both global & local if 1; else, perform only global opt
        Returns:
            model: optimized pyomo model
            optcpu: optimization cpu 
            optsolution: optimal solution 
        ''' 
        # set up pyomo model and define variables 
        model = pe.ConcreteModel()
        model = self.DefineVariable(model)
        
        # if onehotencoding was performed, add dummy constraints
        if self.onehotencoding:
            model = self.dummyconstraints(model)
        
        # set up hidden layers
        model = self.hiddenlayers(model, self.hidden_w_all[0], self.hidden_b_all[0], self.actfunc)
        
        # get slack variable
        sl = self.getslack(slack)
        
        # set up output layer
        model = self.outputlayer(model, self.hidden_w_all[-1], self.hidden_b_all[-1], sl)
        
        # set up graybox constraint
        if self.prob.graycons != None: 
            model = self.knownconstraints(model)

        #perform optimization 
        start = time.time()
        if optall: 
            sol_glob = self.globalopt(model)
            sol_loc = self.multistartlocalopt(model, multistartloc)
        else: 
            sol_glob = self.globalopt(model)
            sol_loc = [] 
        end = time.time()
        
        # combine global and local solution 
        allsolution = [sol_glob] + sol_loc
        
        # df = pd.DataFrame(allsolution)
        # df.to_csv('allsolution.csv')
        optsolution = self.cleansolution(allsolution)
        
        optcpu = end - start
        return model, optcpu, optsolution
    
    def infeasibilityANN(self, multistartloc):
        '''
        Main function to set up infeasibility step for ANN 
        
        Args:
            multistartloc: x sample used for initialization for multistart opt 
        Returns:
            model_sl: pyomo model for infeasibility opt 
            optcpu: optimization cpu 
            optsolution: optimal solution 
        ''' 
        # set up model for infeasibility stage        
        model_sl = pe.ConcreteModel()
        model_sl = self.DefineVariable(model_sl)
        
        # if onehotencoding was performed, add dummy constraints
        if self.onehotencoding:
            model_sl = self.dummyconstraints(model_sl)
        
        # set up hidden layers
        model_sl = self.hiddenlayers(model_sl, self.hidden_w_all[0], self.hidden_b_all[0], self.actfunc)
        
        # for gray box problems 
        if self.prob.graycons != None: 

            model_sl = self.knownconstraints(model_sl)
        
        hidden_w_con = self.hidden_w_all[-1][:,1:]
        hidden_b_con = self.hidden_b_all[-1][1:]
        
        # set up output layer 
        model_sl = self.outputlayer_infeas(model_sl, hidden_w_con, hidden_b_con)
        
        # perform optimization to solve for sl 
        sol_sl = self.globalopt(model_sl)
        
        # get values of slack variables
        sl = [model_sl.sl[i].value for i in range(len(model_sl.sl))]
        # reperform optimization with slack 
        model_sl, optcpu, optsolution = self.optimizeANN(multistartloc, sl, 0)
        
        return model_sl, optcpu, optsolution
    
    
    ''' 
    ===================================================
    Functions used for GP model construction in pyomo 
    ===================================================
    ''' 
    
    def rbfkernel(self, model):
        ''' 
        rbf kernel construction 
        
        Args:
            model: pyomo model
        Returns: 
            model: pyomo model w/ kernel 
        ''' 
    
        model.kind = range(self.xtrain.shape[0]) # kind: number of data points
        model.k = pe.Var(model.kind, within = pe.Reals) # define k for kernel construction
        
        model.kernel = pe.ConstraintList()
        
        for i in range(self.xtrain.shape[0]): 
            # construction of cov matrix for all training points
            model.kernel.add(model.k[i] == pe.exp(-0.5*sum([(model.x[j] - self.xtrain[i,j])**2 for j in range(self.xtrain.shape[1])])))

        return model
        
    def constructGP(self, model, sl):
        '''
        Construct final GP model for constraints and obj 
        
        Args: 
            model: pyomo model
            sl: slack 
        Returns:
            model: pyomo model 
        ''' 
        
        # output index; 1 + # constraint
        model.outind = range(self.OutputDimension)
        model.out = pe.Var(model.outind, within = pe.Reals, initialize = 1)
        
        model.outcon = pe.ConstraintList()
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)
        
        for k in range(self.OutputDimension):
            if k == 0: # for objective     
                model.obj = pe.Objective(expr = sum([self.coef[:,k][i] * model.k[i] for i in range(len(model.k))]))

            else: # for constraints
                f = sum([self.coef[:,k][i] * model.k[i] for i in range(len(model.k))]) - sl[k-1]
                ConType = self.prob.contype[k-1]
                
                if ConType == 'E':
                    model.outcon.add(f <= conrhs[k-1])
                    model.outcon.add(f >= conrhs[k-1])
                elif ConType == 'L':
                     model.outcon.add(f <= conrhs[k-1])
                elif ConType == 'G':
                     model.outcon.add(f >= conrhs[k-1])
        
        return model
    
    def constructGP_infeas(self, model):
        ''' Construct GP for infeasibility step ''' 
        
        # number of outputs
        model.outind = range(self.OutputDimension - 1)
        
        model.outcon = pe.ConstraintList()
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)
        
        # slack 
        model.sl = pe.Var(model.outind, within = pe.Reals, bounds = (0,0.1))
        
        # for constraints
        for k in range(1,self.OutputDimension):
            f = sum([self.coef[:,k][i] * model.k[i] for i in range(len(model.k))]) - model.sl[k-1]
            ConType = self.prob.contype[k-1]
            if ConType == 'E':
                model.outcon.add(f<= conrhs[k-1])
                model.outcon.add(f >= conrhs[k-1])
            elif ConType == 'L':
                 model.outcon.add(f <= conrhs[k-1])
            elif ConType == 'G':
                 model.outcon.add(f >= conrhs[k-1])
        
        # objective - minimize the sum of slack 
        model.obj = pe.Objective(expr = sum([model.sl[k] for k in range(self.OutputDimension-1)]))

        return model
    
    def optimizeGP(self, multistartloc, slack = None, optall = 1):
        
        ''' 
        Main function to construct GP 
        
        Args:
            multistartloc: x location for multistart optimization 
            slack: slack values
            optall: if 1, perform both local and global optimization; else, only local 
        Returns:
            model: pyomo model 
            optcpu: optimization cpu 
            optsolution: optimal solution
        ''' 
        # create pyomo model 
        model = pe.ConcreteModel()
        
        # define variable
        model = self.DefineVariable(model)
        
        # add dummy constraints for one hot encoding         
        if self.onehotencoding:
            model = self.dummyconstraints(model)
        
        # get slack; used after infeasibility stage. else, 0
        sl = self.getslack(slack)

        # set up kernel and construct GP model 
        model = self.rbfkernel(model)
        model = self.constructGP(model, sl)
        
        # for gray box problems, add known constraints
        if self.prob.graycons != None: 
            model = self.knownconstraints(model)
        
        # perform optimization 
        start = time.time()
        if optall: 
            sol_glob = self.globalopt(model)
            sol_loc = self.multistartlocalopt(model, multistartloc)
        else: 
            sol_glob = self.globalopt(model)
            sol_loc = [] 
        end = time.time()

        allsolution = [sol_glob] + sol_loc

        optsolution = self.cleansolution(allsolution)
        
        optcpu = end - start
        return model, optcpu, optsolution
    
    def infeasibilityGP(self, multistartloc):
        '''
        Main function used for GP infeasibility
        
        Args: 
            multistartloc: multistartlocation for local optimization 
        Returns:
            model_sl: infeasibility model
            optcpu: optimization cpu 
            optsolution: optimal solution 
        ''' 
        # construct pyomo model 
        model_sl = pe.ConcreteModel()
        model_sl = self.DefineVariable(model_sl)
       
        # add dummy constraints
        if self.onehotencoding:
            model_sl = self.dummyconstraints(model_sl)
        
        # create kernel and construct GP infeasibility model
        model_sl = self.rbfkernel(model_sl)
        model_sl = self.constructGP_infeas(model_sl)
        
        # for gray-box problems 
        if self.prob.graycons != None: 
            model_sl = self.knownconstraints(model_sl)
        sol_sl = self.globalopt(model_sl)
       
        sl = [model_sl.sl[i].value for i in range(len(model_sl.sl))]
        model_sl, optcpu, optsolution = self.optimizeGP(multistartloc, sl, 0)
        
        return model_sl, optcpu, optsolution
    
    
    ''' 
    ===================================================
    Functions used for SVR model construction in pyomo 
    ===================================================
    ''' 

    def constructSVR(self, model, sl): 
        ''' 
        Function for SVR model construction
        
        Args:
            model: pyomo model
            sl: slack values
        Returns:
            model: pyomo model
        ''' 
        
        model.con = pe.ConstraintList()
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)

        for k in range(self.OutputDimension):
            
            sv = self.sv[k]
            gamma = self.gamma[k]
            kernel = {} 
            for i in range(sv.shape[0]): # construct svs
                kernel[i] = pe.exp(-gamma*sum([(model.x[j] - sv[i,j])**2 for j in range(len(model.x))]))
            
            f = sum([self.coef[k][i] * kernel[i] for i in range(sv.shape[0])]) + self.b[k][0]
                
            if k == 0: #obj 
                model.obj = pe.Objective(expr = f)
            else: # cosntraints
                ConType = self.prob.contype[k-1]
                if ConType == 'L': 
                    model.con.add(f - sl[k-1]<= conrhs[k-1])
                elif ConType == 'G':
                    model.con.add(f + sl[k-1]>= conrhs[k-1])
                elif ConType == 'E':
                    model.con.add(f - sl[k-1]<= conrhs[k-1])
                    model.con.add(f + sl[k-1]>= conrhs[k-1])
                
        return model
    
    def constructSVR_infeas(self, model):
        ''' 
        Function for SVR infeasibility model construction
        
        Args:
            model: pyomo model
        Returns:
            model: pyomo model
        ''' 

        model.outind = range(self.OutputDimension - 1)
        
        model.outcon = pe.ConstraintList()
        conrhs = self.scaleconrhs(self.prob.conrhs, self.prob.yscaler)
        
        model.sl = pe.Var(model.outind, within = pe.Reals, bounds = (0,0.1))
        
        for k in range(1,self.OutputDimension):
            
            sv = self.sv[k]
            gamma = self.gamma[k]
            kernel = {} 
            for i in range(sv.shape[0]): # for svs
                kernel[i] = pe.exp(-gamma*sum([(model.x[j] - sv[i,j])**2 for j in range(len(model.x))]))
            
            f = sum([self.coef[k][i] * kernel[i] for i in range(sv.shape[0])]) + self.b[k][0]
                
            ConType = self.prob.contype[k-1]
            if ConType == 'L': 
                model.con.add(f - model.sl[k-1]<= conrhs[k-1])
            elif ConType == 'G':
                model.con.add(f + model.sl[k-1]>= conrhs[k-1])
            elif ConType == 'E':
                model.con.add(f - model.sl[k-1]<= conrhs[k-1])
                model.con.add(f + model.sl[k-1]>= conrhs[k-1])
        
        # objective -- minimize the sum of slack 
        model.obj = pe.Objective(expr = sum([model.sl[k] for k in range(self.OutputDimension-1)]))

        return model 
            
    def optimizeSVR(self, multistartloc, slack = None, optall = 1):
        
        ''' 
        Main function for SVR optimization 
        
        Args: 
            multistartloc: x location for multistart local optimization
            slack: slack values 
            optall: if 1, perform both global and local optimization
        Returns: 
            model: pyomo model 
            optcpu: optimization cpu 
            optsolution: optimal solution
        
        '''
        # create pyomo model 
        model = pe.ConcreteModel()
        model = self.DefineVariable(model)
                
        # for onehotencoding, add dummy constraints
        if self.onehotencoding:
            model = self.dummyconstraints(model)
        
        sl = self.getslack(slack)

        model = self.constructSVR(model, sl)
        if self.prob.graycons != None:  # for gray-box problems 
            model = self.knownconstraints(model)
        start = time.time()
        if optall: 
            sol_glob = self.globalopt(model)
            sol_loc = self.multistartlocalopt(model, multistartloc)
        else: 
            sol_glob = self.globalopt(model)
            sol_loc = [] 
        end = time.time()
        
        allsolution = [sol_glob] + sol_loc

        optsolution = self.cleansolution(allsolution)
        
        optcpu = end - start
        return model, optcpu, optsolution
    
    def infeasibilitySVR(self, multistartloc):
        '''
        Set up infeasibility model for SVR 
        
        Args:
            multistartloc: x locations for multistart local optimization
        Returns:
            model_sl: infeasibility model
            optcpu: optimization cpu 
            optsolution: optimal solution
        '''
        model_sl = pe.ConcreteModel()
        model_sl = self.DefineVariable(model_sl)
        
        if self.onehotencoding:
            model_sl = self.dummyconstraints(model_sl)
        
        model_sl = self.constructSVR_infeas(model_sl)
        
        if self.prob.graycons != None: 

            model_sl = self.knownconstraints(model_sl)
        
        sol_sl = self.globalopt(model_sl)
       
        sl = [model_sl.sl[i].value for i in range(len(model_sl.sl))]
        model_sl, optcpu, optsolution = self.optimizeSVR(multistartloc, sl, 0)
        return model_sl, optcpu, optsolution


    
    ''' 
    ===================================================
    For gray-box constraints 
    ===================================================
    '''     
    # for gray constraints
    def graycon_transform(self, model):
        ''' 
        Perform automatic transformation for gray-box constraints 
        
        Args:
            model: pyomo model 
        Returns:
            model: pyomo model
        ''' 
        
        # transform original variables for optimization 
        model.graycons = pe.ConstraintList()
        for ind, val in enumerate(self.prob.vartype):
            if val == 'R':
                # scale continuous variables using lb and ub 
                model.graycons.add(model.var[ind] == model.x[ind] * (self.prob.ub[ind] - self.prob.lb[ind]) + self.prob.lb[ind])
                
            else:
                # for binary variable 
                if self.prob.onehotencoding == 1: 
                    # if onehotencoding was performed, transform accordingly 
                    dummyind = range(2*ind - 2, 2*ind)
                    model.graycons.add(model.var[ind] == (1 - model.x[dummyind[0]])/(model.x[dummyind[1]] - model.x[dummyind[0]]))
                    
                    # provide level values to avoid dividing by zero 
                    levelval = random.randint(0,1)
                    model.x[dummyind[0]].value = levelval
                    
                    if levelval == 0:
                        model.x[dummyind[1]] = 1
                    elif levelval == 1:
                        model.x[dummyind[1]] = 0
                
                else: 
                    # if no encoding is performed, simple!  
                    model.graycons.add(model.var[ind] == model.x[ind])
        return model 
    
    
    def graycon_fixbinsol(self, model):
        ''' 
        Function used to fix binary values. This function is only used when 
        MINLP -> NLP (i.e., after binary solutions are determined) 
        
        Args:
            model: pyomo model
        Returns:
            model: pyomo model 
        '''
        
        model.bincon = pe.ConstraintList()
        for ind, sol in enumerate(self.prob.binsol): 
            # manually set a value 
            model.bincon.add(model.x[len(self.prob.contvar) + ind] == sol)
    
        return model 
    
    def knownconstraints(self, model):
        ''' 
        Main function to set up gray-box constraints 
        Args:
            model: pyomo model
        Returns:
            model: pyomo model 
        '''
        
        model.ind = range(len(self.prob.vartype))
        model.var = pe.Var(model.ind)
        
        model = self.prob.graycons(model)
        model = self.graycon_transform(model)
        
        model.pprint()
        if len(self.prob.binsol) > 0:
            model = self.graycon_fixbinsol(model)
        
        return model
    
    
        
    ''' 
    ===================================================
    Other functions for optimization
    ===================================================
    '''
    
    def dummyconstraints(self, model):
        ''' Construct dummy constraints ''' 
        VarType = np.array(self.prob.vartype)
        BinaryInd = np.where(VarType == 'B')[0]
        
        startind = BinaryInd[0]
        
        model.dummy = pe.ConstraintList()
        for i in range(startind, len(model.x), 2):
            model.dummy.add(model.x[i] + model.x[i+1] == 1)
        
        return model 
    
    def selectsolver(self, optglobal):
        ''' 
        Choose optimization solver for GAMS or NEOS
        
        Args: 
            optglobal: if 1, choose a global solver
                else, choose a local solver 
        ''' 
        
        if self.solver == 'gams': # for gams 
            solver_manager = pe.SolverFactory('gams')
            if self.prob.binvar and optglobal:
                options = ['option minlp = baron, optcr = 0.0001;']
            elif self.prob.binvar and not optglobal:
                options = ['option minlp = dicopt;']
            elif not self.prob.binvar and optglobal:
                options = ['option nlp = baron, optcr = 0.0001;']
            elif not self.prob.binvar and not optglobal:
                options = ['option nlp = dicopt;']

    def globalopt(self, model):
        ''' Perform global optimization ''' 
        
        if self.solver == 'gams': # for gams 
            solver_manager = pe.SolverFactory('gams')
            options = ['option minlp = baron, optcr = 0.0001;']
            results = solver_manager.solve(model, keepfiles=True, add_options = options, tmpdir = '.', symbolic_solver_labels= True, warmstart= True)
        else:
            solver_manager = pe.SolverManagerFactory('neos')  # Solve in neos server
            
            if self.prob.binvar: 
                options = ['couenne'] # for minlp opt
            else:
                options = ['lgo'] # for nlp opt 
            
            results = solver_manager.solve(model, opt=options[0])
        
        solution = self.getsolution(model, results)
        
        return solution
    
    def multistartlocalopt(self, model, xloc):
        ''' perform local optimization'''
        if self.solver == 'gams': # for gams 
            solver_manager = pe.SolverFactory('gams')
            if self.prob.binvar:
                options = ['option minlp = dicopt;']
            else:
                options = ['option nlp = conopt;']
        else:
            solver_manager = pe.SolverManagerFactory('neos') # for neos 
            if self.prob.binvar:
                options = ['bonmin'] # for minlp
            else: 
                options = ['ipopt'] # for nlp 
        
        #multistart optimization 

        solution = [] 
        for i in range(xloc.shape[0]): #xloc.shape[0]):
            for j in range(xloc.shape[1]):
                model.x[j].value = xloc[i,j] # provide initial points 
            if self.solver == 'gams': # for gams
                results = solver_manager.solve(model, keepfiles=True, add_options = options, tmpdir = '.', symbolic_solver_labels= True, warmstart= True)#keepfiles=True, add_options = options, tmpdir = '.', symbolic_solver_labels= True) #, opt='bonmin')
            else: # for neos
                results = solver_manager.solve(model, opt=options[0], warmstart=True)
            allsol = self.getsolution(model, results)
            solution.append(allsol)
            
        return solution
    
    def getsolution(self, model, solveresult):
        ''' extract solution from pyomo model ''' 
        xsol = [model.x[i].value for i in range(self.InputDimension)]
        solverstatus = str(solveresult.solver.status)
        modelstatus = str(solveresult.solver.termination_condition)
        
        allsol = xsol + [solverstatus, modelstatus]
        return allsol
    
    def cleansolution(self, solution):
        ''' keep solution with solver status ok 
        
        Args:
            solution: solution obtained from pyomo optimization 
        
        Returns:
            optsolution: clean solution 
        ''' 
        
        moptimal = ['globallyOptimal','locallyOptimal','optimal']
        optsolution = [] 
        for i in solution:
            sstatus = i[-2]
            mstatus = i[-1]
            if mstatus in moptimal: #sstatus is 'ok' and optimal
                sol = i[:-2]
                optsolution.append(sol)
        return optsolution
    
    def getslack(self, slack):
        ''' get slack variable 
        
        Args:
            slack: slack variable; if 0, return a list of 0
        Returns:
            sl: slack values
        ''' 
        if slack: 
            sl = slack 
        else:
            sl = [0] * self.OutputDimension
        return sl
  