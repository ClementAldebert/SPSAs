#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:45:38 2018

@author: koenig.g
"""

###############################################
# Test of different SPSA routines on a Serpent#
# De Mer code. By G.Koenig, the 06/07/2018    #
###############################################


#*************Packages import******************#
#*******The package import******************#

import numpy as np
import Object_SPSA_good_version_06_07 as SPSA # Object for the SPSA
import Serpent_de_Mer_eta_Matricial as seasnake # seasnake code
import matplotlib.pyplot as plt
import scipy.linalg
import timeit

#************Cost_function*********************#

def Cost_func(BC_vec) :
    """ Here we iterate the forward model and compute the cost function."""
    
    # Using the function attribute to import the Seasnake class
    SPM=Cost_func.seasnake_object
    # And the data vector
    Data_vec=Cost_func.data_vec
    
    # Recreating the vector of BC into the corresponding shape
    BC_size=BC_vec.size//2 # Floor division to be sure
    BC_vec=np.array(BC_vec[0:BC_size]+1j*BC_vec[BC_size:],dtype=complex)
    
    # Solving
    SPM.X_eta=scipy.sparse.linalg.spsolve(SPM.A_eta,
                                          np.dot(SPM.B_eta.toarray(),BC_vec))
        # Return the quadratic cost function
    
    return scipy.linalg.norm(SPM.X_eta-Data_vec)

#***********Creating the forward model*********#
    
# Creating and initializing the seasnake_data
SPM_obj=seasnake.SerpentdeMerMatricial_eta() 

# Linking this object with the cost function
Cost_func.seasnake_object=SPM_obj

# Getting the data    
SPM_obj.set_grid_manually(20,20,dx=200.,dy=200.)

# Setting BC's
SPM_obj.sol_Kelvin_wave()
SPM_obj.eta[1:-1,1:-1]=0.

#Getting the data
SPM_obj.create_bc_vector_overelevation() # Y vector
SPM_obj.create_solving_matrix_overelevation() # A matrix
SPM_obj.create_projection_matrix_overelevation() # Projection of boundary

# Solving 

Cost_func.data_vec=scipy.sparse.linalg.spsolve(SPM_obj.A_eta,
                        np.dot(SPM_obj.B_eta.toarray(),SPM_obj.Y_eta)) 
# Here are the Data

#************Declaration of variables**********#
#************For the SPSA**********************#
args={'Y':np.zeros(SPM_obj.Y_eta.size*2),'cost_func':Cost_func,
      'tol':1e-8,'n_iter':100,'gamma':0.101,'alpha':0.602,'A':6e3,
      'comp':False,'anim':False} 
# Arguments for the SPSA

Test_SPSA=SPSA.SPSA(**args) # We initialize it

Y_size=Test_SPSA.Y.size//2

# Dictionnaries for looping on the different SPSA's

List_SPSA=['SPSA vanilla','SPSA one-sided','SPSA mean','SPSA momentum',
           'SPSA adaptative step','SPSA stochastic directions',
           'SPSA stochastic directions with momentum']

List=['SPSA vanilla']
SPSA_methods={'SPSA vanilla':Test_SPSA.SPSA_vanilla,
              'SPSA one-sided':Test_SPSA.SPSA_one_sided,
                'SPSA mean':Test_SPSA.SPSA_mean,
                'SPSA momentum':Test_SPSA.SPSA_momentum,
                'SPSA adaptative step':Test_SPSA.SPSA_adaptative_step_size,
                'SPSA stochastic directions':Test_SPSA.SPSA_stochastic_direction,
                'SPSA stochastic directions with momentum':Test_SPSA.SPSA_stochastic_direction_momentum}

SPSA_arg={'SPSA vanilla':None,'SPSA one-sided':None,'SPSA mean':2,
                'SPSA momentum':None,'SPSA adaptative step':2.,
                'SPSA stochastic directions':20,
                'SPSA stochastic directions with momentum':20}
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
    Test_SPSA.c*=SPM_obj.Y_eta.mean().real/10.
    Test_SPSA.a*=SPM_obj.Y_eta.mean().real/200.
    
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


#************TOTAL INVERSION PROCEDURE*********#

# We used to cost_func.data_vec cause this is were we store the original good 
# Solution. If we used X_eta we will have the solution computed at the last
# SPSA iteration

t1=timeit.default_timer()
Y_inv=scipy.sparse.linalg.lsqr(SPM_obj.B_eta,SPM_obj.A_eta*Cost_func.data_vec)
Y_inv=Y_inv[0] # So that I do not save the additional values it gives me
# copying time
SPSA_exec_time['Total inverse']=timeit.default_timer()-t1
# We have to modify it a little to get something that can fit in the cost
# Function
Cost_func_inv=Cost_func(np.hstack([Y_inv.real,Y_inv.imag])) 
# To know if the total inversion worked well
# Adding the total inverse formulation
List_SPSA.append('Total inverse')
SPSA_J_values['Total inverse'] = Cost_func_inv*np.ones(args['n_iter'])
SPSA_Y_values['Total inverse'] = Y_inv
SPSA_exec_model['Total inverse'] = 1
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