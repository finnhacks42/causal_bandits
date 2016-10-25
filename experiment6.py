# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:19:07 2016

@author: finn
"""

from models import ScaleableParallelConfounded,ParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,AlphaUCB,ThompsonSampling
from experiment_config import Experiment
import numpy as np



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
           

experiment = Experiment(6)
experiment.log_code()
                
N = 50
N1 = 1
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .3
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

simulations = 10000

model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)

T_vals = range(25,626,25)

algorithms = [GeneralCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)
