#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:45:38 2018

@author: koenig.g
"""

#################################################
# Test of different SPSA routines on a KPP code #
# By C. Aldebert, the 09/07/2018                #
#################################################

plt.close('all')


#*************Packages import******************#
#*******The package import******************#
import sys
import os
#sys.path.append('../SPSA_lib') # For SPSA
sys.path.append('../DERNIERE_VERSION_VALIDE') # For SPSA
import numpy as np
import Object_SPSA_good_version_06_07 as SPSA # Object for the SPSA
import matplotlib.pyplot as plt
import scipy.linalg
import timeit

#************Cost_function*********************#

def Cost_func(theta) :
    """ Here we iterate the forward model and compute the cost function."""
    
    # reference data
    u_ref = np.loadtxt("../KPP_steadysurface/Data/u_steady_ref.txt")[1,]    # chope la 2eme ligne

    # save new parameters
    np.savetxt("../KPP_steadysurface/param_surf2df.txt",theta.real)

    # call model run
    os.system("../KPP_steadysurface/run_steady.exe")

    # model results
    u = np.loadtxt("../KPP_steadysurface/RES/u_steady.txt")[1,]     # chope la 2eme ligne
    print(u)    
    
    # difference between results
    y = np.subtract(u_ref,u)
    # return sum of square of errors
    return scipy.linalg.norm(y)    



#************Declaration of variables**********#
#************For the SPSA**********************#
args={'Y':np.ones(2),'cost_func':Cost_func,
      'tol':1e-8,'n_iter':10000,'gamma':0.101,'alpha':0.602,'A':1000,
      'comp':False,'anim':True} 
# Arguments for the SPSA

Test_SPSA=SPSA.SPSA(**args) # We initialize it

Y_size=Test_SPSA.Y.size//2

# Dictionnaries for looping on the different SPSA's

List_SPSA=['SPSA vanilla','SPSA one-sided','SPSA mean','SPSA momentum',
           'SPSA adaptative step','SPSA accelerating step',
           'SPSA stochastic directions','SPSA stochastic directions with momentum']

List=['SPSA vanilla']
SPSA_methods={'SPSA vanilla':Test_SPSA.SPSA_vanilla,
              'SPSA one-sided':Test_SPSA.SPSA_one_sided,
                'SPSA mean':Test_SPSA.SPSA_mean,
                'SPSA momentum':Test_SPSA.SPSA_momentum,
                'SPSA adaptative step':Test_SPSA.SPSA_adaptative_step_size,
                'SPSA accelerating step':Test_SPSA.SPSA_accelerating_step_size,
                'SPSA stochastic directions':Test_SPSA.SPSA_stochastic_direction,
                'SPSA stochastic directions with momentum':Test_SPSA.SPSA_stochastic_direction_momentum}

SPSA_arg={'SPSA vanilla':None,'SPSA one-sided':None,'SPSA mean':2,
                'SPSA momentum':None,'SPSA adaptative step':2.,
                'SPSA accelerating step':2.,
                'SPSA stochastic directions':2,
                'SPSA stochastic directions with momentum':2}
# Here we store the arguments for the various SPSA, we do not use dictionnary
# In dictionnary yet. But it may become useful
    
SPSA_Y_values={}
SPSA_J_values={} #Empty dictionnaries to store data
SPSA_exec_time={}
SPSA_exec_model={} # To record the number of calls to the cost function

# Now looping on it  
for ln in List_SPSA:
    # Reinitializing
    Test_SPSA.__init__(**args)
    # Setting gain parameters
    Test_SPSA.c*=1e-5
    Test_SPSA.a*=1e-3
    
    # Counting execution time
    t1=timeit.default_timer()
    
    if SPSA_arg[ln] is None :
        SPSA_methods[ln]() # Calling SPSA procedure
    else :
        SPSA_methods[ln](SPSA_arg[ln])
    #Saving execution time
    SPSA_exec_time[ln]=timeit.default_timer()-t1
    # Saving the number of calls to the cost function
    SPSA_exec_model[ln]=Test_SPSA.k_costfun
    #Testing
    # Saving data
    SPSA_J_values[ln]=Test_SPSA.J

    SPSA_Y_values[ln]=np.array(Test_SPSA.Y[0:Y_size]+1j*Test_SPSA.Y[Y_size:],dtype=complex)


#*********Now we can plot************************#
# We use dictionnary and list to store variables, it will be me easier to use
# Than having a lot of times the same instruction
# Comparing solution
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
# Plotting the cost functions
fig_J=plt.figure()
ax_J=fig_J.add_subplot(1,1,1)
# Plotting an histogram for execution time
fig_time=plt.figure()
ax_time=fig_time.add_subplot(1,1,1)
# And another histogram for the number of calls of the model
fig_mod=plt.figure()
ax_mod=fig_mod.add_subplot(1,1,1)
ctr=0 # For position of bars
for ln in List_SPSA:
    ax.plot(np.hstack([SPSA_Y_values[ln].real,SPSA_Y_values[ln].imag]),
            marker='*',label=ln)
    # Here we plot each result and we concatenate the real and imaginary part
    ax_J.plot(SPSA_J_values[ln],label=ln)
    # Here we plot the cost function on another graph
    ax_time.bar(ctr,SPSA_exec_time[ln])
    # And for the numbers of call to the model
    ax_mod.bar(ctr,SPSA_exec_model[ln])
    ctr+=1 # Incrementing the counter
    
ax.legend(loc='best')
ax.set_xlabel('# Of component of the bc vector')
ax.set_ylabel('Value (arbitrary unit)')
# Data figure
ax_J.legend(loc='best')
ax_J.set_yscale('log')
ax_J.set_xlabel('Iteration')
ax_J.set_ylabel('Cost function')
# Cost function figure
ax_time.set_xticklabels(['0']+List_SPSA)
ax_time.set_ylabel('Execution time (s)')
# Cost function figure
ax_mod.set_xticklabels(['0']+List_SPSA)
ax_mod.set_ylabel('# executions of the model')

fig.show()
fig_J.show()
fig_time.show()
fig_mod.show()