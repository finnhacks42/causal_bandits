# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:52:59 2016

@author: finn
"""
import numpy as np
from algorithms import GeneralCausal,ParallelCausal,SuccessiveRejects, AlphaUCB,ThompsonSampling
from models import Parallel
from experiment_config import Experiment, now_string

def regret_vs_T(model,algorithms,T_vals,simulations = 10):
    
    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    
    for T_indx,T in enumerate(T_vals): 
        for a_indx,algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
        print T
                
    return regret
           
experiment = Experiment(3)
experiment.log_code()
                  
simulations = 10000
N = 50
m = 2
epsilon = .3
model = Parallel.create(N,m,epsilon)
T_vals = range(10,6*model.K,25)
algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()


experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)



