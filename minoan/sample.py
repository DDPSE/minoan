# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:42:02 2020

@author: k3148
"""
import numpy as np
import pyDOE as pD
from sklearn.preprocessing import OneHotEncoder 
import math as m
import itertools as it

class sampling: 
    
    ''' 
    Main class used for minlp and nlp sampling 
    
    Args:
        prob: contains problem information 
    
    '''
    
    def __init__(self, prob):
        ''' 
        Constructs all necessary attributes 
        '''
        self.simulator = prob.simulator
        self.contvar = prob.contvar
        self.binvar = prob.binvar
        self.dim = len(self.contvar) + len(self.binvar)
        self.lb = prob.lb
        self.ub = prob.ub
        self.noutput = len(prob.contype) + 1
        self.onehotencoding = prob.onehotencoding
        self.binsol = prob.binsol
        self.contype = prob.contype
        self.conrhs = prob.conrhs
    
    def minlp_nlhs(self):
        ''' Set the number of LHS required for each discrete level '''
        nlevel = 2**len(self.binvar)
        nlhs = m.ceil(int(10 * self.dim + 1)/nlevel) 
        return max(5,nlhs)
    
    def nlp_nlhs(self):
        ''' Set the number of LHS required for NLP (10n+1) ''' 
        nlhs = 10 * len(self.contvar) + 1
        return nlhs
    
    def binarylevels(self):
        '''
        MINLP: find all combinations of binary variables
        
        Returns:
            binlevels: all possible combinations of binvary variables    
        
        ''' 
        bin_values = {}
        for i in self.binvar:
            bin_values[i] = np.array([0,1])
        binlevels = list(it.product(*(bin_values[i] for i in self.binvar)))  
        return binlevels
    
    ''' for nlp ''' 
    def LHS_nlp(self, nlhs):
        ''' 
        NLP: perform lhs for given dimension and nlhs 
        
        Args: 
            nlhs: desired number of lhs points 
        

        Returns:
            xlhs: initial LHD 
        ''' 
        dim = len(self.contvar)
        xlhs = pD.lhs(dim, samples = nlhs)
        return xlhs
    
    def LHS_minlp(self, nlhs):
        ''' 
        MINLP: perform lhs for given dimension and nlhs 
        
        Args: 
            nlhs: desired number of lhs points
        
        Returns: 
            xlhs: initial LHD
        '''
        
        binlevels = self.binarylevels()
        xlhs = np.zeros((0,self.dim))
        for l in binlevels:
            lhs_nlp = self.LHS_nlp(nlhs)
            lhs_bin = np.round(np.array([list(l)]*nlhs),0)
            lhs_level = np.hstack((lhs_nlp, lhs_bin))
            xlhs = np.concatenate((xlhs, lhs_level), axis = 0)
        return xlhs
    
    def create(self, condropind = [], convio = 0, xonly = 0):
        ''' 
        Main function to create an intial sample set
        
        Args:
            condropind: used only during MINLP -> NLP search to drop unnecessary constraint
            convio: used only during MINLP -> NLP search to store con vio 
            xonly: if 1, do not call the simulation; just return x 
        
        Returns:
            xlhs: initial LHD; unscaled using original bounds 
            ynew: simulation output
            conrhs: constraint rhs; only important during MINLP -> NLP 
            convio: constraint violation; only important during MINLP -> NLP 
            condropind: index of constraint that needs to be dropped
                used only when all values are the same after bin values are fixed 
        ''' 

        
        if not self.binvar: #nlp sampling
            nlhs = self.nlp_nlhs()
            xlhs0 = self.LHS_nlp(nlhs)
        else: # minlp sampling
            nlhs = self.minlp_nlhs()
            xlhs0 = self.LHS_minlp(nlhs)
        
        # unscale xlhs 
        xscaler = createscaler(self.lb, self.ub)
        xlhs = xscaler.inverse_transform(xlhs0)
        
        if xonly: 
            return xlhs
        
        # call simulation
        ylhs = simulate(xlhs, self.simulator, self.binsol)
        
        # clean output
        ynew, conrhs, contype, convio, condropind = self.checkoutput(ylhs, condropind, convio)
        
        return xlhs, ynew, conrhs, contype, convio, condropind

    def augmentLHS(self, x,number_new_pts):    
        '''
        Augment LHS
        Used when no feasible solution is found for multiple consecutive iterations 
        
        Args: 
            x: initial lhd 
            number_new_pts: number of new points to be added 
        
        Returns:
            new points: augmented lhd (new points)
        '''
        dim = x.shape[1]
        number_old_pts = x.shape[0]
        number_cells = (number_old_pts+number_new_pts)**2
        
        
        cell_size = 1./(number_cells+1)    
        cell_lo = [0+i*cell_size for i in range(number_cells+1)]
        cell_up = [(i+1)*cell_size for i in range(number_cells+1)]
        
        candidate_pts = []
        number_candidate_pts = number_new_pts*2
        row_vec = np.argsort(np.random.uniform(0,1,number_old_pts))
        col_vec = np.argsort(np.random.uniform(0,1,dim))
        for i in col_vec:
            candidate_cells = list(range(number_cells+1))
            for j in row_vec:
                lo = x[j,i]-cell_lo
                filled = np.max(np.where(lo>=0))
                if filled in candidate_cells:
                    candidate_cells.remove(filled)        
            candidate_cells = np.random.choice(candidate_cells,number_candidate_pts)
            candidate_pts.append([np.random.uniform(cell_lo[k],cell_up[k],1) for k in candidate_cells])
        
        candidate_pts1 = np.reshape(candidate_pts,(dim,number_candidate_pts)).T
        new_pts = []    
        old_points = x
        for k in range(number_new_pts): 
            distance = [np.min(np.sum((old_points-candidate_pts1[i,:])**2,1)) for i in range(len(candidate_pts1))]
            selected = np.argmax(distance)
            new_pts.append(candidate_pts1[selected,:])
            old_points=np.concatenate((old_points,np.array([candidate_pts1[selected,:]])),axis=0)
            candidate_pts1=np.delete(candidate_pts1,selected,0)        
        new_pts = np.array(new_pts)
        return new_pts
    
    def augment(self, xs, encoder, npts=2):
        ''' 
        Main function to augment lhd 
        For MINLP, augment LHD per level 
        
        Args:
            xs: scaled x
            encoder: one-hot encoder; used when self.onehotencoding = 1
            npts: desired number of new points 
            
        Returns: 
            xnew: augmented dataset 
        '''
        if self.binvar: # for MINLP problem
            
            if self.onehotencoding:
                # if one hot encoding = 1, first decode x 
                xs = self.decode(xs, encoder)
            
            # find binary levels and augment xs per level 
            binlevels = self.binarylevels()
            xnew = np.zeros((0,self.dim))
            for ind, val in enumerate(binlevels):
                xlevel = xs[(xs[:,len(self.contvar):] == val).all(axis = 1)]
                xslevel = xlevel[:,:len(self.contvar)]
                xnewcont = self.augmentLHS(xslevel, npts)
                xnewbin = np.round(np.array([list(val)]*npts),0)
                xnewlevel = np.hstack((xnewcont, xnewbin))
                xnew = np.concatenate((xnew, xnewlevel), axis = 0)
        else: # for NLP
            xnew = self.augmentLHS(xs, npts)
        return xnew
    
    def encode(self, x):
        
        ''' 
        Perform one hot encoding for MINLP 
        
        Args:
            x: input x 
        
        Returns:
            encoder: sklearn encoder
            xencoded: encoded x 
        ''' 
        
        encoder = OneHotEncoder()
        
        xtoencode = x[:,len(self.contvar):]
        df = encoder.fit_transform(xtoencode).toarray()
        
        xencoded = np.hstack((x[:,:len(self.contvar)], df))
        return encoder, xencoded
    
    def decode(self, x, encoder):
        ''' 
        Reverse one-hot encoding for MINLP 
        
        Args: 
            x: input dx
            encoder: sklearn encoder
        Returns:
            xdecoded: decoded x
        ''' 
        xbin = x[:,len(self.contvar):]
        xbin_decode = encoder.inverse_transform(xbin)
        
        xcont = x[:,:len(self.contvar)]
        xdecoded = np.hstack((xcont, xbin_decode))
        return xdecoded
        
        
    def preprocess(self, x, y, onehotencoding = 1):
        ''' 
        Preprocess the data by: 
            1) scaling input and output data using bounds
            2) perform one hot encoding for MINLPs 
    
        Args: 
            x: input data
            y: output data 
            onehotencoding: if 1, perform one-hot encoding 
        Returns: 
            xs: scaled input data
            ys: scaled output data
            xscaler: class of xscaler 
            yscaler: class of yscaler 
            encoder: sklearn encoder for one-hot encoding 
        ''' 
        # scale input data using original lb and ub       
        xscaler = createscaler(self.lb, self.ub)
        xs = xscaler.fit_transform(x)
        
        if len(y) > 0:
            # scale y using minmax scaler 
            yscaler = createscaler(y.min(axis = 0), y.max(axis = 0))
            ys = yscaler.fit_transform(y)
        else:
            # return empty data frame if len(y) = 0
            ys = [] 
            yscaler = [] 
        
        if onehotencoding: 
            # perform onehotencoding 
            encoder, xs = self.encode(xs)
        else:
            encoder = None
        return xs, ys, xscaler, yscaler, encoder
    
    def findbinarycon(self, y):
        ''' 
        Calculate constraint violation for constraints that only contain binary
        variables (binary constraint). This function is used only during 
        MINLP -> NLP after fixing binary variables. If a constraint only has 
        binary variables, the value is the same regardless of the value of continuous variable 
        
        Args: 
            y: output data
        Returns:
            condropind: index of constraint with no change in value 
            convio: binary constraint violation
        ''' 
        condropind = [] 
        convio = 0
        ycon = y[:,1:]
        for col in range(ycon.shape[1]):
            selectedcol = ycon[:,col]
            val = np.unique(selectedcol) 
            # if all values are the same, it is a binary constraint
            if len(val) == 1:
                condropind.append(col + 1)
                
                # calculate constraint violation 
                diff = val - self.conrhs[col]
                if self.contype[col] == 'E':
                    vio = abs(diff)
                elif self.contype[col] == 'L':
                    vio = max(diff,0)
                elif self.contype[col] == 'G':
                    vio = abs(min(diff,0))
                
                convio += vio
        return condropind, convio 
    
    def checkoutput(self, y, condropind, convio):
        ''' 
        Clean y to remove binary constraints; only used for MINLP -> NLP 
        
        Args: 
            y: data output
            condropind: index of binary constraint
            convio: constraint violation of binary constraint
        Returns:
            ynew: clean y 
            conrhs: clean conrhs
            contype: clean contype
            convio: constraint violation
            condropind: index of binary constraint 
        ''' 
        
        if not condropind:
            condropind, convio = self.findbinarycon(y)
        
        if condropind:
            # if binary constraint exist, drop the corresponding column from ynew
            ynew = np.delete(y, condropind, axis = 1)
            # drop conrhs and contype of binary constraint
            conrhs = np.delete(self.conrhs, np.array(condropind) - 1).tolist()
            contype = np.delete(self.contype, np.array(condropind) - 1).tolist()
        else:
            ynew = y
            conrhs = self.conrhs
            contype = self.contype
        return ynew, conrhs, contype, convio, condropind

def keepsamplesbin(x, y, binsol):
    ''' 
    When transitioning from MINLP -> NLP, keep some of the collected samples 
    with binary variables = binary solution. This reduces sampling requirement
    
    Args: 
        x: all collected x during MINLP search
        y: all collected y during MINLP search
        binsol: obtained binary solution 
    Returns:
        xtokeep: x where data = binary solution; return only continuous x
        ytokeep: y where data = binary solution
        
    '''
    ncont = x.shape[1] - len(binsol)
    
    # keep samples if values = binsol
    xtokeep = x[(x[:,-len(binsol):] == binsol).all(axis = 1)][:,:ncont]
    ytokeep = y[(x[:,-len(binsol):] == binsol).all(axis = 1)]
    
    return xtokeep, ytokeep

def keepsamplesbounds(x,y,lb,ub):
    ''' 
    When transitioning from NLP -> NLP zoom in, keep some of the collected samples
    with values within the new lb and ub. This reduces sampling requirement 
    
    Args: 
        x: all collected x during NLP1 search 
        y: all collected y during NLP1 search
        lb: new lb 
        ub: new ub 
    Returns:
        xtokeep: x where data within new bounds
        ytokeep: y where data within new bounds 
    '''
    xtokeep = x[(x>= lb).all(axis = 1) & (x <= ub).all(axis = 1)]
    ytokeep = y[(x>= lb).all(axis = 1) & (x <= ub).all(axis = 1)]
    return xtokeep, ytokeep
        
    
def simulate(x, f, binval):
    ''' 
    Simulator 
    
    Args:
        x: simulation input
        f: simulation 
        binval: if nonempty, fix binary values
    Returns: 
        yall: simulation output 
        
    '''
    # if binval != empty, fix binary values 
    # used only for minlp -> nlp 
    if len(binval) > 0: 
        df = [list(binval)]*x.shape[0]
        x = np.concatenate((x, df), axis = 1)
    
    yall = []
    
    for i in range(x.shape[0]):
        y = f(x[i])
        yall.append(y)    
    return np.array(yall)      

class createscaler:
    
    ''' 
    Create a class for scaler 
    
    Args: 
        lb: lower bound
        ub: upper bound
    ''' 
    def __init__(self, lb, ub):
        self.data_min_ = np.array(lb)
        self.data_max_ = np.array(ub)
    
    def fit_transform(self, data):
        ''' 
        Scale the data (minmaxscaler)
        
        Args:
            data: unscaled data in original domain 
        Returns: 
            datascaled: scaled data 
        ''' 
        data = np.array(data)
        datascaled = (data - self.data_min_)/(self.data_max_ - self.data_min_)
        return datascaled
    
    
    def inverse_transform(self, datatounscale):
        ''' 
        Unscale the data 
        
        Args: 
            datatounscale: data to be unscaled
        Returns:
            dataunscaled: unscaled data 
        ''' 
        datatounscale = np.array(datatounscale)
        dataunscaled = datatounscale * (self.data_max_ - self.data_min_) + self.data_min_
        return dataunscaled

    

#%%    
