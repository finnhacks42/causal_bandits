# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from models import ParallelConfounded,ScaleableParallelConfounded
from algorithms import SuccessiveRejects,GeneralCausal,AlphaUCB,ThompsonSampling,ParallelCausal
from experiment_config import Experiment
import numpy as np


def regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = 1000): 
    m_vals = []
    models = []
    regret = np.zeros((len(algorithms),len(N1_vals),simulations))

    for m_indx,N1 in enumerate(N1_vals):
        model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1,compute_m = False)
        eta = [0,0,1.0/(N1+2.0),0,0,0,1-N1/(N1+2.0)]
        model.compute_m(eta_short = eta)
       
        print N1,model.m
        m_vals.append(model.m)
        models.append(model)
        for a_indx, algorithm in enumerate(algorithms):
            for s in xrange(simulations):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
                
    
    return m_vals,regret,models
    
   
experiment = Experiment(4)
experiment.log_code()
    
N = 50
N1_vals = range(1,N,3)
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .3
simulations = 10000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal(),AlphaUCB(2),ThompsonSampling()]


epsilon = .3
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

m_vals,regret,models = regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = simulations)
experiment.plot_regret(regret,m_vals,"m",algorithms,legend_loc = "lower right",legend_extra = [ParallelCausal])








