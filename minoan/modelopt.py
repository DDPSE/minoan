# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:28:15 2020

@author: k3148
"""
import numpy as np
import time
from pyomo_opt import *
import math as m
from sample import *
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

class modelopt:
    ''' 
    Main class for surrogate modeling and optimization 
    
    Args:
        prob: class with problem information 

    ''' 
    
    def __init__(self, prob):
        self.prob = prob 

    def cvsplit(self, xs):
        ''' 
        Calculate cross-validation fraction based on nsample.
        For a smaller dataset, use a smaller fraction 
        
        Args: 
            xs: data input 
        Returns:
            testfrac: fraction of test set 
        ''' 
        if xs.shape[0] < 50:
            testfrac = 0.2
        else:
            testfrac = 0.3
        return testfrac
    
    def annfitting(self, xs, ys, testfrac):
        '''
        Construct a neural network model
        
        Args:
            xs: scaled input data
            ys: scaled output data
            testfrac: fraction of test set for cv 
        Returns:
            bestann: dictionary that contains the best model and model info 
        ''' 
        
        # fix the number of hidden nodes 
        nhidden = m.ceil(xs.shape[1]*(2/3) + ys.shape[1])
        nhiddennode = [(nhidden,)]
        
        # alternatively, can try out multiple #nodes 
        # NHidden = np.arange(10,100,10) # for relu
        # nhiddennode = [(i,) for i in nhidden]
        
        # for hyperparameter tuning
        param_grid = {'activation': ['tanh'],
                 'solver':['lbfgs'],
                 'alpha' : [0.0001],
                 'hidden_layer_sizes': nhiddennode,
              }
        
        model_start = time.time() 
        
        TrainError = [] 
        TestError = [] 
        BestModel = {} 
        
        for j in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size = testfrac)
            
            Model = MLPRegressor(early_stopping = False, max_iter=1000, validation_fraction = 0.3)
            Grid = GridSearchCV(Model, param_grid, scoring='neg_mean_squared_error', cv = 5, verbose=0, n_jobs = -1)
            Grid.fit(X_train, Y_train)
            BestModel[j] = Grid.best_estimator_
            
            YP_train = BestModel[j].predict(X_train)
            YP_test = BestModel[j].predict(X_test)
            
            TrainError.append(mean_squared_error(Y_train, YP_train))
            TestError.append(mean_squared_error(Y_test, YP_test))
        
        ModelCPU = time.time() - model_start 
        
        # choose the model with min test error
        BestModelInd = np.argmin(TestError)
        
        SelectedModel = BestModel[BestModelInd]
        
        bestann = {'Model': SelectedModel, \
                 'TrainError' : TrainError[BestModelInd], \
                     'TestError' : TestError[BestModelInd], \
                         'ModelCPU' : ModelCPU
                     } 
        return bestann 
    
    def gpfitting(self, xs, ys, testfrac):
        '''
        Construct a Gaussian Process model
        
        Args:
            xs: scaled input data
            ys: scaled output data
            testfrac: fraction of test set for cv 
        Returns:
            bestgp: dictionary that contains the best model and model info 
        ''' 
        model_start = time.time() 
        
        TrainError = [] 
        TestError = [] 
        BestModel = {} 
        
        for j in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size = testfrac)
            
            BestModel[j] = GaussianProcessRegressor(alpha = 1e-10, normalize_y = False)
            BestModel[j].fit(X_train, Y_train)
            
            YP_train = BestModel[j].predict(X_train)
            YP_test = BestModel[j].predict(X_test)
            
            TrainError.append(mean_squared_error(Y_train, YP_train))
            TestError.append(mean_squared_error(Y_test, YP_test))
        
        ModelCPU = time.time() - model_start 
        
        # choose the model with min test error 
        BestModelInd = np.argmin(TestError)
        
        SelectedModel = BestModel[BestModelInd]
        
        bestgp = {'Model': SelectedModel, \
                 'TrainError' : TrainError[BestModelInd], \
                     'TestError' : TestError[BestModelInd], \
                         'ModelCPU' : ModelCPU
                     } 
        return bestgp 
   
    def svrfitting(self, xs, ys, testfrac):
        '''
        Construct a SVR model
        
        Args:
            xs: scaled input data
            ys: scaled output data
            testfrac: fraction of test set for cv 
        Returns:
            bestsvr: dictionary that contains the best model and model info 
        ''' 
        param_grid = {'C':[1e0, 1e1, 1e2, 1e3], 'gamma': np.logspace(-5,2,5)}
        
        model_start = time.time()
        
        noutput = ys.shape[1]
        BestModel = {} 
        CVTrainError = []
        CVTestError = [] 
        
        for j in range(10):
            BestModel[j] = {} 
            
            X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size = testfrac)
            TrainError = []
            TestError = []
            for i in range(noutput):
                model = SVR()
                Grid = GridSearchCV(model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 0)
                Grid.fit(X_train, Y_train[:,i])
                BestModel[j][i] = Grid.best_estimator_
                
                YP_train = BestModel[j][i].predict(X_train)
                YP_test = BestModel[j][i].predict(X_test)
                
                TrainError.append(mean_squared_error(Y_train[:,i], YP_train))
                TestError.append(mean_squared_error(Y_test[:,i], YP_test))
            
            AvgTrainError = np.average(TrainError)
            AvgTestError = np.average(TestError)

            CVTrainError.append(AvgTrainError)
            CVTestError.append(AvgTestError)
        
        ModelCPU = time.time() - model_start
        BestModelInd = np.argmin(CVTestError)
        SelectedModel = BestModel[BestModelInd]
        
        bestsvr = {'Model': SelectedModel, \
                 'TrainError' : CVTrainError[BestModelInd], \
                     'TestError' : CVTestError[BestModelInd], \
                         'ModelCPU' : ModelCPU
                     } 
        return bestsvr

    def fit(self, xs, ys):
        ''' 
        Main function for model fitting 
        
        Args: 
            xs: scaled input data
            ys: scaled output data
        Returns:
            bestmodel: best model & model info 
        '''
        
        
        modeltype = self.prob.modeltype
        testfrac = self.cvsplit(xs)
        
        
        if modeltype == 'ANN':
            bestmodel = self.annfitting(xs, ys, testfrac)
        elif modeltype == 'GP':
            bestmodel = self.gpfitting(xs, ys, testfrac)
        elif modeltype == 'SVR':
            bestmodel = self.svrfitting(xs, ys, testfrac)
        
        return bestmodel 
    
    def optimize(self, bestmodel, multistartloc):
        ''' 
        Main class for model optimization:
            For a selected model type, perform optimization. If no feasible 
            solution is found, enter the infeasibility stage to find the 
            most feasible solution. If no feasible solution is found, return 
            empty list optsol. 
        
        Args:
            bestmodel: best model inherited from self.fit
            multistartloc: multistart optimization location for local search 
        Returns:
            optsol: optimal solution 
            optcpu: optimization computation time 
        '''
            
        
        optsol = [] 
        optcpu = [] 
        opt = pyomoopt(self.prob, bestmodel)

        if self.prob.modeltype == 'ANN':
            model, optcpu, optsol = opt.optimizeANN(multistartloc, slack = None)
            if not optsol: # enter infeasibility
                model, optcpu, optsol = opt.infeasibilityANN(multistartloc)
        
        elif self.prob.modeltype == 'GP':
            model, optcpu, optsol = opt.optimizeGP(multistartloc, slack = None)
            if not optsol:  # enter infeasibility
                model, optcpu, optsol = opt.infeasibilityGP(multistartloc)
        
        elif self.prob.modeltype == 'SVR':
            if not optsol: # enter infeasibility
                model, optcpu, optsol = opt.infeasibilitySVR(multistartloc)
            
        return optsol, optcpu
    
    def fitoptimize(self, xs, ys, xloc):
        '''
        Construct a model and optimize it until:
            1) a feasible solution is found
            2) if no feasible solution is found, solve infeasibility problem
        If neither 1 nor 2 is satisfied after 5 iterations, augment LHS 
        
        Args:
            xs: scaled input data
            ys: scaled output data
            xloc: multi start location for local optimization 
        ''' 

        solfound = False
        count = 0
        # repeat searches for 5 iterations         
        while not solfound and count < 5:
            # fit and optimization a model 
            bestmodel = self.fit(xs, ys)
            optsol, optcpu = self.optimize(bestmodel, xloc)
            if optsol:
                solfound = True
            count += 1
        
        # if no solution is found, augmentLHS
        if not solfound:
            solunique = sampling(self.prob).augment(xs, self.prob.encoder)
        else: 
            # remove duplicate solutions
            solutions = np.round(np.array(optsol), 5)
            solunique = np.unique(solutions, axis = 0)
    
        return solunique
    