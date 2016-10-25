# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:30:34 2016

@author: finn
"""
import numpy as np
from math import sqrt,ceil
from models import Parallel
from algorithms import GeneralCausal, ParallelCausal, SuccessiveRejects, AlphaUCB, ThompsonSampling
from experiment_config import now_string, Experiment


def regret_vs_T_vary_epsilon(model,algorithms,T_vals,simulations = 10):

    regret = np.zeros((len(algorithms),len(T_vals),simulations))
    
    for T_indx,T in enumerate(T_vals): 
        print T
        epsilon = sqrt(model.K/(a*T))
        model.set_epsilon(epsilon)
        for s in xrange(simulations):
            for a_indx, algorithm in enumerate(algorithms):
                regret[a_indx,T_indx,s] = algorithm.run(T,model)
        
    return regret
    
experiment = Experiment(2)
experiment.log_code()

N= 50
simulations = 10000
a = 9.0
m = 2
model = Parallel.create(N,m,.1)

Tmin = int(ceil(4*model.K/a))
Tmax = 10*model.K
T_vals = range(Tmin,Tmax,100)

algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret = regret_vs_T_vary_epsilon(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)




