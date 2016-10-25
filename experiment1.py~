# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:47:47 2016

@author: finn
"""
import numpy as np
from models import Parallel
from algorithms import GeneralCausal, ParallelCausal, SuccessiveRejects,AlphaUCB,ThompsonSampling
from experiment_config import Experiment


def regret_vs_m(algorithms,m_vals,N,T,epsilon,simulations = 10):  
    models = []    
    regret = np.zeros((len(algorithms),len(m_vals),simulations))
    for m_indx,m in enumerate(m_vals):
        model = Parallel.create(N,m,epsilon)
        models.append(model)
        print "built model {0}".format(m)
        for s in xrange(simulations):
            for a_indx,algorithm in enumerate(algorithms):
                regret[a_indx,m_indx,s] = algorithm.run(T,model)
    
    return regret,models

experiment = Experiment(1)
experiment.log_code()
          
# Experiment 1
N = 50
epsilon = .3
simulations = 10000
T = 400
algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]
m_vals = range(2,N,2)
    
regret,models = regret_vs_m(algorithms,m_vals,N,T,epsilon,simulations = simulations)


experiment.plot_regret(regret,m_vals,"m",algorithms,legend_loc="lower right")
experiment.log_state(globals())


    