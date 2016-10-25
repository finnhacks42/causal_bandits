# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:44:51 2016

Compare the performance of the algorithms on a Confounded Parallel bandit for which 
there is no action to set Z. The Parallel algorithm is mis-specified in this setting, as it assumes there are no confounders.
If the resulting bias exceeds epsilon then the Parallel algorithm will never identify the best arm.
"""


from models import ParallelConfoundedNoZAction
from algorithms import SuccessiveRejects,GeneralCausal,ParallelCausal,AlphaUCB,ThompsonSampling
from experiment_config import Experiment
import numpy as np
from itertools import chain



def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    pulls = np.zeros((len(algorithms),len(T_vals),model.K),dtype=int)
    for T_indx,T in enumerate(T_vals): 
        
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
                if algorithm.best_action is not None:
                    pulls[a_indx,T_indx,algorithm.best_action] +=1
        print T
                
    return regret,pulls


experiment = Experiment(5)
experiment.log_code()
           
simulations = 10000                 

q = (.2,.8,.7,.3)
pz = 0.6 
pY = np.asarray([[ 0 , 1  ],[1 , 0]])

N0 = 6
N1 = 1
N2 = 14
N = N0+N1+N2

q10,q11,q20,q21 = q
pXgivenZ0 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q10),np.full(N2,q20)))    
pXgivenZ1 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q11),np.full(N2,q21)))
pXgivenZ = np.stack((np.vstack((1.0-pXgivenZ0,pXgivenZ0)),np.vstack((1.0-pXgivenZ1,pXgivenZ1))),axis=2) # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
pYfunc = lambda x: pY[x[N0],x[N-1]]
model = ParallelConfoundedNoZAction(pz,pXgivenZ,pYfunc)

T_vals = list(chain(range(10,100,25),range(85+50,450,50),range(435+100,1036,100)))

algorithms = [GeneralCausal(),ParallelCausal(),SuccessiveRejects(),ThompsonSampling(),AlphaUCB(2)]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)

print "m",model.m

    


