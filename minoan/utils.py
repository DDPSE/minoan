# -*- coding: utf-8 -*-
"""
MINOAN: DDPSE Lab @ GT 
Copyright (C) 2020 - Sophie Kim 
"""
import numpy as np
import scipy.stats as ss
from joblib import Parallel, delayed


class info:
    
    ''' 
    Class used to store all problem information (e.g., dimension, lb, ub, etc)
    
    Args: 
        vartype: variable type; R for continuous and B for binary variable
        lb: variable lower bound
        ub: variable upper bound
        contype: constraint type; E for equality constraint, L for <= and G for >= constraint
        conrhs: right hand size of all constraints; real number
        simulator: black-box simulation 
        modeltype: surrogate model type; ANN, GP, or SVR 
        onehotencoding: 1 for construction of mixed-integer model with one hot encoding 
        nprocs: number of processors; used for parallel search 
        maxeval: number of maximum samples allowed; used for termination
        solver: optimization solver; if specified 'gams', gams solver will be called 
            else, if gams is not available, neos server will be called 
            
    Returns:
        class with all problem information 
                
    ''' 
    
    def __init__(self, vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, solver):

        self.vartype = vartype
        self.lb = lb
        self.ub = ub 
        self.contvar, self.binvar = self.variable()
        self.graycons = graycons
        self.simulator = simulator
        self.modeltype = modeltype
        self.onehotencoding = onehotencoding
        self.nprocs = nprocs
        self.maxeval = maxeval 
        self.contype = contype
        self.conrhs = conrhs 
        self.solver = solver
        self.ncons = len(self.conrhs)
        self.dim = len(self.contvar) + len(self.binvar)
    
    def variable(self):
        ''' 
        Set up variable name:
            if 'R', it is a continuous variable
            if 'B', it is a binary variable 
            
        Returns: 
            continuous: list that contains continuous variables
            binary: list that contains binary variables
        ''' 
        continuous = [] 
        binary = [] 
        for ind, val in enumerate(self.vartype):
            if val == 'R':
                continuous.append('x' + str(ind+1))
            else:
                binary.append('x' + str(ind + 1))
        return continuous, binary
    

class ranksol: 
    ''' 
    Main class for find the best incumbent solution by:      
        1) Least constraint violation
        2) Lowest objective value 
        3) Combined score of 1 and 2 
    
    Args:
        prob: class that contains problem information
        xopt: x optimal solution
        yopt: y optimal solution

    ''' 
    def __init__(self, prob, xopt, yopt):
        self.prob = prob 
        self.xopt = xopt
        self.yopt = yopt

    def bestobjind(self):
        ''' 
        Choose the solution with minimum objective value
        
        Returns:
            bestsolind: index of best solution
        ''' 
        allobjs = self.yopt[:,0]
        bestsolind = np.argmin(allobjs)
        return bestsolind
    
    def findconvio(self, ysol):
        ''' 
        Calculate constraint violations in the original domain 
        
        Args:
            ysol: matrix that contains all output values (obj & constraints)
        
        Returns:
            convio: constraint violation 
        ''' 
        
        ysol = ysol.reshape(-1, len(self.prob.conrhs)+1)
        allcons = ysol[:,1:].copy() # drop objective column 
        
        convio = np.array([])
        for i in range(allcons.shape[0]):
            # calculate constraint violation 
            diff = allcons[i] - self.prob.conrhs
            
            for ind, val in enumerate(self.prob.contype):
                if val == 'L': # for <= constraint
                    diff[ind] = max(diff[ind],0)
                elif val == 'G': #for >= constraint
                    diff[ind] = abs(min(diff[ind],0))
                elif val == 'E': #for equality constraint
                    diff[ind] = abs(diff[ind])
            
            convio = np.concatenate((convio, diff), axis = 0)
            
        return convio.reshape(-1,allcons.shape[1])
        
    def scaleconvio(self, convio):
        '''
        Constraint violation in the scaled domain
        To scale, max and min of y are used 
        Useful especially when outputs have drastically different magnitude 
        
        Args: 
            convio: constraint violation 
        
        Returns:
            convioscaled: scaled constraint violation 
        '''
        convioscaled = convio.copy()
        for i in range(convio.shape[0]):
            for j in range(convio.shape[1]):
                vio = convio[i,j]
                if vio:
                    # scale convio 
                    smin = self.prob.yscaler.data_min_[j+1]
                    smax = self.prob.yscaler.data_max_[j+1]
                    convioscaled[i,j] = (vio - smin) / (smax - smin)
                    
        return convioscaled

    def bestconind(self):
        ''' 
        Choose the best incumbent solution as minimum constraint violation 
        
        Returns:
            bestsolind: index of best incumbent solution
        ''' 
        
        # calculate and scale constraint violation 
        convio = self.findconvio(self.yopt)
        convioscaled = self.scaleconvio(convio)
        
        # sum up constraint violation in the original and scaled domain 
        consum = convio.sum(axis = 1)
        consumscaled = convioscaled.sum(axis = 1)
        
        # choose the best solution as min of scaled constraint violation 
        bestsolind = np.argwhere(consumscaled == consumscaled.min())
        
        if len(bestsolind) > 1: 
            # if multiple feasible solutions exist, choose the one with min obj 

            objvalues = self.yopt[bestsolind,0]
            bestsolind = bestsolind[objvalues.argmin()]
        
        return bestsolind.tolist()[0]
        
    def bestallind(self):
        ''' 
        Choose the best incumbent solution with minimum score. 
        Score is calculated by taking account of both obj value and constraint violation 
        
        Returns:
            bestsolind: index of best incumbent solution 
        '''
        
        # calculate constraint violation and scale it 
        convio = self.findconvio(self.yopt)
        convioscaled = self.scaleconvio(convio)
        
        # sum up con vio 
        consum = convio.sum(axis = 1)
        consumscaled = convioscaled.sum(axis = 1)
        
        obj = self.yopt[:,0]
        
        # calculate solution score 
        vscore = ss.rankdata(consumscaled) # rank data by con vio
        objscore = ss.rankdata(obj) # rank data by objective value 
        totalscore = np.add(vscore, objscore)/2 # average two scores
        
        # choose the solution with min score 
        bestsolind = np.argwhere(totalscore == totalscore.min())
        if len(bestsolind) > 1:
            # if tie exists, choose the one with the smallest constraint violation
            selected = consumscaled[bestsolind]
            bestsolind = bestsolind[np.argmin(selected)]
        
        return bestsolind.tolist()[0]


    def bestsolmain(self, scorecrit):
        '''
        Main class to find the best incumbent solution 
        
        Args:
            scorecrit: criterion for solution selection
        Returns:
            xbest: best x solution
            ybest: best y solution
            vbest: constraint violation corresponding to xbest and ybest
        
        '''
        if scorecrit == 1: 
            # choose the solution with minimum constraint violation
            bestsolind = self.bestconind()
        elif scorecrit == 2:
            # choose the solution with minimum obj value
            bestsolind = self.bestobjind()
        elif scorecrit == 3: 
            # choose the solution with minimum solution score 
            bestsolind = self.bestallind() 
        
        xbest = self.xopt[bestsolind].flatten()
        ybest = self.yopt[bestsolind]
        
        # calculate the constraint violation of the best solution 
        vbest = self.findconvio(ybest).sum()
        
        return xbest, ybest, np.array([vbest])

def parallelsearch(parallel_input, nproc, nlpsearch):   
    ''' 
    Perform parallel search for multiple promising binary solutions 
    
    Args: 
        parallel_input: dictionary that contains each search information 
        nproc: number of processors 
        nlpsearch: function that performs nlp search; inherited from class main
    Returns:
        resultall: dictionary that contains returns solution obtained from 
            all parallel searches 
    ''' 
    
    resultall = Parallel(n_jobs=nproc)(delayed(nlpsearch)(parallel_input[i]) for i in range(len(parallel_input)))    
    return resultall 

def checktermination(prob, ysol, vio, nsample, modelerr):
    '''
    Check whether the algorithm satisfied one of the criteria and terminate: 
        1) no improvement in the objective value over 10 consecutive iterations
        2) constraint violation <= 1e-5, model error <= 1e-5, and length of solution is > 5
        3) number of sample collected exceeds max allowed number of samples
    
    Args:
        prob: class that contains problem information
        ysol: all y solutions 
        vio: all constraint violation 
        nsample: current # sample
        modelerror: model error 
    
    Returns:
        conv: convergence status; 1 for convergence, 0 for no convergence
        stat: convergence status that corresponds to criteria above 
    ''' 
    
    # initialize
    stat = 0 
    conv = False
    obj = ysol[:,0]
    vio = vio
    maxsample = prob.maxeval
    
    # calculate objective improvement 
    if any(obj == 0): 
        dobj = [abs(obj[i+1] - obj[i]) for i in range(len(obj)-1)]
    elif len(obj) == 0: 
        # first iteration 
        dobj = 100 # set a large number
    else: 
        dobj = [abs((obj[i+1] - obj[i])/obj[i+1]) for i in range(len(obj)-1)]
    
    # check termination condition 
    nsol = 10
    if all(np.array(dobj[-nsol:]) < 0.01) and len(obj) > nsol:  #criterion 1
        stat = 1
    elif vio[-1] < 1e-5 and modelerr < 1e-5 and len(vio) >= 5: #criterion 2
        stat = 2
    elif nsample > maxsample: # criterion 3
        stat = 3
    
    if stat > 0: 
        conv = True
    
    return conv, stat

def reducebounds(xsol, lb, ub, radius = 0.01): 
    
    ''' 
    Reduce problem bound by a given radius (aka zoom in)
    
    Args:
        xsol: x solution to zoom in 
        lb: original lb
        ub: original ub
        radius: radius used for zooming in 
    Returns:
        lbupdated: new lb
        upupdated: new ub 
    ''' 
    
    # choose new domain 
    domain = np.array([abs((ub[i] - lb[i])*radius) for i in range(len(lb))])
    lb0 = xsol - domain
    ub0 = xsol + domain
    # new lb cannot be smaller than the original lb 
    lbupdated = [max(lb[i], lb0[i]) for i in range(len(lb))]
    # new ub cannot be greater than the original ub 
    ubupdated = [min(ub[i], ub0[i]) for i in range(len(ub))]
    return lbupdated, ubupdated
