# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:51:06 2020

@author: k3148
"""
import numpy as np
import pandas as pd
import time
import os
import random
import warnings

from minoan.modelopt import *
from minoan.sample import *
from minoan.utils import *
from minoan.pyomo_opt import *

class Minoanopt: 
    
    ''' 
    Main class for MINOAN 
    Args:
        vartype (list): variable type; R for continuous and B for binary 
        lb (list): variable lower bound
        ub (list): variable upper bound
        contype (list): constraint type; E for equality constraint and G or L for inequality constraint
        conrhs (list): constraint right hand side (e.g., g(x)>=conrhs)
        graycons (function): granconstraint if known; pyomo model 
        simulator (function): black-box simulation
        modeltype (str): surrogate model type; ANN, SVR, GP, or hybrid
        onehotencoding (int): 1 for MI model construction; 0 for relaxed model construction
        nprocs (int): number of processors; if nprocs > 1, parallel processing is used
        maxeval (int): maximum number of simulation evaluation allowed
        solver (str): 'gams' for gams interface; 'neos' for neos server
        put (int): 1 to save intermediate results in csv; else 0

    MINOAN has 6 main steps. These steps are the same for MINLP and NLP 
    1) read problem information 
    2) perform initial sampling and scale the data (sample.py)
    3) perform surrogate modeling and optimization (modelopt.py)
    4) unscale xopt; inquire simulation to obtain yopt 
    5) calculate constraint violation; find the best incumbent solution 
    6) add the sample; perform steps 3-5 again 
    
    ''' 
    
    def __init__(self, vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs = 1, maxeval = 5000, solver = 'gams', put = 0):
        self.prob = info(vartype, lb, ub, contype, conrhs, graycons, simulator, modeltype, onehotencoding, nprocs, maxeval, solver)
        self.convio = 0
        self.condropind = []
        self.prob.binsol = [] # initialize binary solution 
        self.dfsol = pd.DataFrame()
        self.nsample = 0
        self.xall = np.array([]).reshape(0,self.prob.dim)
        self.yall = np.array([]).reshape(0,len(self.prob.conrhs) + 1)
        self.scorecrit = 3
        self.hybrid = 0 # initialize hybrid status 
        self.cwd = os.getcwd()
        self.put = put # 1 to print result as csv file
        self.probcheck() 
    
    def probcheck(self):
        
        if len(self.prob.lb) != len(self.prob.ub):
            warnings.warn('Length of lower bound != length of upper bound. Please check the bounds')
        
        if len(self.prob.ub) != len(self.prob.vartype):
            warnings.warn('Length of variable type != length of lower bound')
        
        if len(self.prob.ub) != len(self.prob.vartype):
            warnings.warn('Length of variable type != length of lower bound')
        
        if len(self.prob.contype) != len(self.prob.conrhs):
            warnings.warn('Constraint info length does not match. Please check contype and conrhs')
        
        
     
    def itersol(self, x, y, scorecrit):
        ''' 
        Rank all intermediate solutions and return the best solution
        
        Args:
            x: all optimal x solutions
            y: all optimal y solutions
            scorecrit: solution selection criterion
        '''
        
        # rank and find the best solution
        r = ranksol(self.prob, x, y)
        xsol, ysol, vio = r.bestsolmain(scorecrit)
        
        self.xsol = np.vstack((self.xsol, xsol))
        self.ysol = np.vstack((self.ysol, ysol))
        self.vio = np.vstack((self.vio, vio))
        return 
    
    
    def multistartloc(self):
        ''' create sample set for multistart optimization initialization '''
        s = sampling(self.prob)
        xloc = s.create(self.condropind, self.convio, xonly = 1)
        
        xloc, yloc, xscaler, yscaler, encoder = s.preprocess(xloc, [], self.prob.onehotencoding)
        return xloc
    
    def fit_optimize(self, xs, ys):
        ''' 
        Surrogate modeling and optimization 
        Args:
            xs: scaled x input
            ys: scaled y output
        
        Returns:
            xopt: optimal x solution 
            yopt: optimal y solution
        ''' 
        # get sites for multistart initialization
        xloc = self.multistartloc()
        
        # fit and optimize model 
        opt = modelopt(self.prob)
        xsopt = opt.fitoptimize(xs, ys, xloc)
        
        # if onehotencoding = 1, decode 
        if self.prob.onehotencoding and xsopt.shape[1] != self.prob.dim:
            s = sampling(self.prob)
            xsopt = s.decode(xsopt, self.prob.encoder)
            
        # clean the solution; unscale xopt and inquire simulation 
        xopt = self.prob.xscaler.inverse_transform(xsopt)
        yopt = simulate(xopt, self.prob.simulator, self.prob.binsol)
        
        # drop col if coldropind exists; used after initial MINLP stage 
        if self.condropind: 
            yopt = np.delete(yopt, self.condropind, axis = 1)
        return xopt, yopt
    
    def run(self):
        '''
        Main function for surrogate-based optimization. This class performs 
        sampling, surrogate modeling, and optimization until convergence. 
        
        Returns:
            stat: convergence status; 1 for convergence; 0 else
            xbest: best optimal solution for x
            ybest: best optimal solution for y
            viobest: constraint violation of the best solution
            nsample: # samples      
       
        '''
        # initial LHD 
        s = sampling(self.prob)
        x, y, self.prob.conrhs, self.prob.contype, self.convio, self.condropind = s.create(self.condropind, self.convio)
        self.nsample += x.shape[0]

        
        if self.condropind: 
            # drop column(s) from y if self.condropind is not empty 
            # used only for MINLP -> NLP 
            self.yall = np.delete(self.yall, self.condropind, axis = 1)
            
        self.xall = np.vstack((self.xall, x))
        self.yall = np.vstack((self.yall, y))
        
        # initialize solution arrays
        self.xsol = np.array([]).reshape(0,x.shape[1])
        self.ysol = np.array([]).reshape(0,y.shape[1])
        self.vio = np.array([]).reshape(0,1)
        
        
        i = 0
        converge = False 

        while not converge: 
            
            # preprocess the data - scale & onehotencoding 
            xs, ys, xscaler, yscaler, encoder = s.preprocess(self.xall, self.yall, self.prob.onehotencoding)
        
            self.prob.xscaler = xscaler
            self.prob.yscaler = yscaler
            self.prob.encoder = encoder 
            
            # fit surrogate model and optimize 
            xopt, yopt = self.fit_optimize(xs, ys)
            
            if self.prob.solver == 'neos':
                os.system('rm *log')
                os.system('rm *nl')
                os.system('rm *sol')
            
            # add xopt and yopt to sample set 
            self.xall = np.vstack((self.xall, xopt))
            self.yall = np.vstack((self.yall, yopt))
            
            self.nsample += xopt.shape[0]
            
            xiter = np.vstack((self.xsol, xopt))
            yiter = np.vstack((self.ysol, yopt))
                        
            # find the best solution so far among self.xsol and self.ysol
            self.itersol(xiter, yiter, self.scorecrit)
            
            # check termination 
            converge, stat = checktermination(self.prob, self.ysol, self.vio, self.nsample, 2)
        
            xbest = np.array(self.xsol[-1])
            ybest = np.array([self.ysol[-1][0]])
            viobest = self.vio[-1]
            
            self.cpu = time.time() - self.start
            
            # to save all intermediate results 
            xall = np.append(self.xsol[-1], self.prob.binsol)
            colnames = ['x' + str(i) for i in range(len(xall))] + ['obj'] + ['vio'] + ['nsample'] + ['cpu'] + ['stat'] + ['term']
            dfsol = pd.DataFrame(np.concatenate((xall, ybest, viobest, np.array([self.nsample]), np.array([self.cpu]), np.array([stat]), np.array([self.terminate]))), colnames).T
            self.dfsol = pd.concat([self.dfsol, dfsol], axis = 0)

            if self.put:
                self.dfsol.to_csv('solution.csv')

        return stat, xbest, ybest, viobest, self.nsample
    
    
    def minlptonlp(self, xbest):
        ''' 
        Transition from MINLP to NLP search by using binary solutions 
        
        Args:
            xbest: best x solution determined from MINLP stage
        Returns:
            uniquebinsol: all promising binary solutions
        '''
        
        self.prob.binsol = xbest[-len(self.prob.binvar):]
        
        self.xall, self.yall = keepsamplesbin(self.xall, self.yall, self.prob.binsol)
        
        self.prob.lb = self.prob.lb[:len(self.prob.contvar)]
        self.prob.ub = self.prob.ub[:len(self.prob.contvar)]
        
        uniquebinsol = self.findallbinsols()
        
        self.prob.binvar = [] 
        self.prob.onehotencoding = 0
        # self.prob.binsol = xbest[len(self.prob.contvar):]
        return uniquebinsol
    
    def parallelinputs(self, uniquebinsol):
        '''
        Create input dictionary for parallel search 
        
        Args:
            uniquebinsol: all promising binary solutions
        Returns:
            parallel_input: dictionary that contains unique binary solutions for 
                parallel search 
        ''' 
       
        parallel_input = {} 
        for i in range(len(uniquebinsol)):
            binsol = uniquebinsol[i]
            parallel_input[i] = binsol
            
        return parallel_input
                
    def findallbinsols(self):
        ''' find all unique binary solutions ''' 
        allbinsol = self.xsol[:,-len(self.prob.binvar):]
        uniquebinsol = np.unique(allbinsol, axis = 0)
        return uniquebinsol
    
    def cleanparallelsol(self, solutions):
        ''' 
        Clean parallel search solutions
        
        Args:
            solutions: parallel search results
        Returns:
            stat: convergence status
            xbest: best x solution
            ybest: best y solution
            vio: constraint violation of best solution
            nsample: # samples
        '''
        y0 = [solutions[i][2] for i in range(len(solutions))]
        
        nsample_all = [solutions[i][4] for i in range(len(solutions))]
        
        y = np.array(y0)
        bestind = y.argmin()
        
        stat, xbest, ybest, vio, nsample = solutions[bestind]
        
        self.prob.binsol = xbest[len(self.prob.contvar):]
        xbest = xbest[:len(self.prob.contvar)]
        
        # sum up all samples collected during parallel search; add this to total nsample
        self.nsample += np.sum(nsample_all)
        return stat, xbest, ybest, vio, self.nsample
            
    def nlpsearch(self, binsol):
        #update problem information for minlp -> nlp
        ''' 
        Update problem information for minlp -> nlp search 
        Used only for parallel binary search
        
        Args: 
            binsol: values that binary variables will be fixed at
        Returns:
            stat: convergence status
            xbest: x best solution
            ybest: y best solution
            viobest: constraint violation 
            nsample: # samples
        ''' 
        # create a directory name for parallel search randomly
        dirname = random.randint(1,999)
        
        if not os.path.exists(str(dirname)):
            os.mkdir(str(dirname))
        
        os.chdir(str(dirname))
        
        # initialize
        self.nsample = 0
        self.prob.binsol = binsol

        stat, xbest, ybest, viobest, nsample = self.run()
        
        xbest = np.concatenate((xbest, binsol))
        
        os.chdir(self.cwd)
        
        return stat, xbest, ybest, viobest, nsample
    
    def nlptonlp2(self, xbest):
        ''' 
        After NLP search, zoom in with reduced bound to further improve the solution 
        
        Args:
            xbest: best x solution
        ''' 
        # update score criterion to choose a solution with smallest con vio 
        self.scorecrit = 1
        # reduce bound
        self.prob.lb, self.prob.ub = reducebounds(xbest, self.prob.lb, self.prob.ub)
        # keep samples that belong in the reduced bound 
        self.xall, self.yall = keepsamplesbounds(self.xall, self.yall, self.prob.lb, self.prob.ub)
        return 

    def main(self):
        
        '''
        Main class for the entire search 
        Returns:
            stat: convergence status; 1 for convergence; 0 else
            xbest: best x solution
            ybest: best y solution 
            viobest: constraint violation of the best solution
            nsample: total # samples
        '''
        
        self.start = time.time()
        
        # if self.prob.binvar: 
        #     self.minlp = 1
        # else:
        #     self.minlp = 0
        
        if self.prob.modeltype == 'hybrid':
            self.prob.modeltype = 'ANN'
            self.hybrid += 1 # update hybrid status 
        
        self.terminate = 1
        # perform initial search 
        stat, xbest, ybest, viobest, nsample = self.run()

        # if MINLP, fix binary solution and perform NLP search 
        if self.prob.binvar: 
            
            if self.hybrid:
                self.prob.modeltype = 'GP'
            
            self.terminate += 1
            
            uniquebinsol = self.minlptonlp(xbest)
            
            # if multiple promising binary solutions exist, perform multiple search in parallel
            if self.prob.nprocs > 1 and len(uniquebinsol) > 1:    
                # create parallel inputs and perform parallel search 
                parallel_input = self.parallelinputs(uniquebinsol)
                resultall = parallelsearch(parallel_input, self.prob.nprocs, self.nlpsearch)   
                stat, xbest, ybest, viobest, nsample = self.cleanparallelsol(resultall)
            
            else:
                # else, perform search for only the best binary solution 
                stat, xbest, ybest, viobest, nsample = self.run()

        self.terminate += 1
        
        # reduce variable bounds for solution refinement 
        self.nlptonlp2(xbest)
        stat, xbest, ybest, viobest, nsample = self.run()
    
        xbest = np.concatenate((xbest, self.prob.binsol))
        
        return stat, xbest, ybest, viobest
        
      