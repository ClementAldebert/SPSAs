#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:49:16 2018

@author: koenig.g
"""

###############################################
# Test of different SPSA routines on standard #
# problems, in 2 dimensions to get an idea of #
# The way they move                           #
###############################################

plt.close('all')

#*************Packages import******************#
#*******The package import******************#
import sys
import os
sys.path.append('../SPSA_lib') # For SPSA libraries

import numpy as np
import Object_SPSA_Modified_version_12_07 as SPSA # Object for the SPSA
import matplotlib.pyplot as plt
import scipy.linalg
import timeit,random

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


def Cost_func_2D(x,y) :
    """ Here we use the KPP model"""
    theta=np.zeros(2)
    theta[0]=x
    theta[1]=y
    
    # reference data
    u_ref = np.loadtxt("../KPP_steadysurface/Data/u_steady_ref.txt")[1,]    # chope la 2eme ligne

    # save new parameters
    np.savetxt("../KPP_steadysurface/param_surf2df.txt",theta)

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
# Random init for KPP
Init_point=np.ones(2)
# So that all methods start at the same point
args={'Y': Init_point,'cost_func':Cost_func,'tol':1e-8,'n_iter':100,
      'gamma':0.101,'alpha':0.602,'A':10,'comp':False,'anim':False
      ,'save_Y':True,'n_Y':20 } 


Test_SPSA=SPSA.SPSA(**args) # We initialize it

# Dictionnaries for looping on the different SPSA's

List_SPSA=['SPSA vanilla','SPSA one-sided','SPSA mean','SPSA momentum',
           'SPSA adaptative step',
           'SPSA accelerating step',
           'SPSA stochastic directions',
           'SPSA stochastic directions with momentum',
           'SPSA stochastic directions with RMS hessian']
SPSA_methods={'SPSA vanilla':Test_SPSA.SPSA_vanilla,
              'SPSA one-sided':Test_SPSA.SPSA_one_sided,
                'SPSA mean':Test_SPSA.SPSA_mean,
                'SPSA momentum':Test_SPSA.SPSA_momentum,
                'SPSA adaptative step':Test_SPSA.SPSA_adaptative_step_size,
                'SPSA accelerating step':Test_SPSA.SPSA_accelerating_step_size,
                'SPSA stochastic directions':Test_SPSA.SPSA_stochastic_direction,
                'SPSA stochastic directions with momentum':Test_SPSA.SPSA_stochastic_direction_momentum,
                'SPSA stochastic directions with RMS hessian':Test_SPSA.SPSA_stochastic_RMS_prop}

SPSA_arg={'SPSA vanilla':None,'SPSA one-sided':None,'SPSA mean':{'n_mean':2},
                'SPSA momentum':None,'SPSA adaptative step':{'mult_size':2.},
                'SPSA accelerating step':{'mult_size':2.},
                'SPSA stochastic directions':{'batch_size':1},
                'SPSA stochastic directions with momentum':{'batch_size':1},
                'SPSA stochastic directions with RMS hessian':
                    {'batch_size':1,'mom_coeff':0.9}}
SPSA_markers={'SPSA vanilla':'*','SPSA one-sided':'p','SPSA mean':'o',
                'SPSA momentum':'v','SPSA adaptative step':'8',
                'SPSA accelerating step':'8',
                'SPSA stochastic directions':'x',
                'SPSA stochastic directions with momentum':'D',
                'SPSA stochastic directions with RMS hessian':'h'}
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
    # Parameters for KPP
    Test_SPSA.c*=1e-5
    Test_SPSA.a*=1e-3


    
    # Counting execution time
    t1=timeit.default_timer()
    if SPSA_arg[ln] is None :
        SPSA_methods[ln]() # Calling SPSA procedure
    else :
        SPSA_methods[ln](**SPSA_arg[ln])
    #Saving execution time
    SPSA_exec_time[ln]=timeit.default_timer()-t1
    # Saving the number of calls to the cost function
    SPSA_exec_model[ln]=Test_SPSA.k_costfun
    #Testing
    # Saving data
    SPSA_J_values[ln]=Test_SPSA.J

    SPSA_Y_values[ln]=Test_SPSA.Y_his
    SPSA_Y_values[ln]=np.vstack([SPSA_Y_values[ln],Test_SPSA.Y]) # We save the values
#*********Now we can plot************************#
# First we generate a map of our 2D functions and its value
fig_dir=plt.figure()
ax_dir=fig_dir.add_subplot(1,1,1)
#XX,YY=np.meshgrid(np.arange(-2.,2.,.01),np.arange(-2.,2.,.01))
#ZZ=XX.copy()
#for i in range(0,np.size(ZZ,0)):
#    for j in range(0,np.size(ZZ,1)):
#        ZZ[i,j]=Cost_func_2D(XX[i,j],YY[i,j])
#        
#np.savetxt("../KPP_steadysurface/costfunc_mapXX.txt",XX)
#np.savetxt("../KPP_steadysurface/costfunc_mapYY.txt",YY)
#np.savetxt("../KPP_steadysurface/costfunc_mapZZ.txt",ZZ)

XX=np.loadtxt("../KPP_steadysurface/costfunc_mapXX.txt")
YY=np.loadtxt("../KPP_steadysurface/costfunc_mapYY.txt")
ZZ=np.loadtxt("../KPP_steadysurface/costfunc_mapZZ.txt")
        
        
cax=ax_dir.contourf(XX,YY,np.log10(ZZ))
cbar=plt.colorbar(cax)
cbar.set_label('Value of the function')
# We use dictionnary and list to store variables, it will be me easier to use
# Than having a lot of times the same instruction
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
    ax_dir.plot(SPSA_Y_values[ln][:,0],SPSA_Y_values[ln][:,1],
                marker=SPSA_markers[ln],linestyle='solid',label=ln)
    # Plotting on the evolution map
    ax_J.plot(SPSA_J_values[ln],label=ln)
    # Here we plot the cost function on another graph
    ax_time.bar(ctr,SPSA_exec_time[ln])
    # And for the numbers of call to the model
    ax_mod.bar(ctr,SPSA_exec_model[ln])
    ctr+=1 # Incrementing the counter

# Legend for the map
ax_dir.scatter(SPSA_Y_values['SPSA vanilla'][0,0],
    SPSA_Y_values['SPSA vanilla'][0,1],marker='o',color='red',
    s=400,label='Starting point')
ax_dir.legend(loc='best')

# Data figure
ax_J.legend(loc='best')
try :
    ax_J.set_yscale('log')
except :
    pass
ax_J.set_xlabel('Iteration')
ax_J.set_ylabel('Cost function')
# Cost function figure
ax_time.set_xticklabels(['0']+List_SPSA)
ax_time.set_ylabel('Execution time (s)')
# Cost function figure
ax_mod.set_xticklabels(['0']+List_SPSA)
ax_mod.set_ylabel('# executions of the model')

fig_dir.show()
fig_J.show()
fig_time.show()
fig_mod.show()
