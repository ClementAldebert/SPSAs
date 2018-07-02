#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:48:05 2018

@author: koenig.g
"""

##############################################
# Code to test SPSA methods on data generated#
# By the Serpent de Mer code. By G.Koenig,the#
# 26/06/2018                                 #
# Modified the 27/06/2018 to implement       #
# Plotting and other                         #
# Modification to transform complex vectors  #
# Into real vectors , the 27/06/2018         #
# Modified the 28/06/2018 to test animation  #
# Function                                   #
# Modified the 29/06/2018 to include a total #
# Inversion scheme                           #
# Modified the 29/06/2018 to test a mean     #
# Formulation                                #
# Modified the 02/07/2018 to test a momentum #
# Formulation                                #
##############################################


#*************Packages import******************#
#*******The package import******************#
import sys
sys.path.append('../SPSA_lib') # For SPSA
sys.path.append('../Serpent_de_Mer') # For Serpent de Mer

import numpy as np
import Object_SPSA_02_07_Momentum as SPSA # Object for the SPSA
import Serpent_de_Mer_eta_Matricial as seasnake # seasnake code
import matplotlib.pyplot as plt
import scipy.linalg

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

#***********Using the SPSA*********************#

#************Declaration of variables**********#

args={'Y':np.zeros(SPM_obj.Y_eta.size*2),'cost_func':Cost_func,
      'tol':1e-19,'n_iter':60000,'gamma':0.101,'alpha':0.602,'A':6e3,
      'comp':False,'anim':False} 
# Arguments for the SPSA

Test_SPSA=SPSA.SPSA(**args) # We initialize it

# We modifiy c to have something closer to what we expect
Test_SPSA.c*=SPM_obj.Y_eta.mean().real/10.

# A modifications for covariance, everything is symmetric here
Test_SPSA.a[1,20]=1. # Lower left
Test_SPSA.a[20,1]=1.
Test_SPSA.a[18,21]=1. # Lower right
Test_SPSA.a[21,18]=1.
Test_SPSA.a[54,57]=1. # Upper left
Test_SPSA.a[57,54]=1.
Test_SPSA.a[55,74]=1. # Upper right
Test_SPSA.a[74,55]=1. # Upper right

Test_SPSA.a[77,96]=1. # Lower left
Test_SPSA.a[96,77]=1.
Test_SPSA.a[94,97]=1. # Lower right
Test_SPSA.a[97,94]=1.
Test_SPSA.a[130,133]=1. # Upper left
Test_SPSA.a[133,130]=1.
Test_SPSA.a[131,150]=1. # Upper right
Test_SPSA.a[150,131]=1. # For imaginary part only

# I took advantage of the fact that my Poisson operator in the present problem
# I entirely real, so I not have to care about the real part/ imaginary part 
# Coupling that might have arisen.
# For the size order
Test_SPSA.a*=SPM_obj.Y_eta.mean().real/200.

# Modifying it to remove the corner values that are not to influence the
# Solution

# C modifications
Test_SPSA.c[0]=1e12
Test_SPSA.c[19]=1e12
Test_SPSA.c[56]=1e12
Test_SPSA.c[75]=1e12

#################################################
#          SPSA PROCEDURE                       #
#################################################
Test_SPSA.SPSA_momentum(mom_coeff=0.) # Momentum estimator

# Here the output vector will be a real vector twice as long. We then have
# To modify it a little bit
Y_size=Test_SPSA.Y.size//2
Y=np.array(Test_SPSA.Y[0:Y_size]+1j*Test_SPSA.Y[Y_size:],dtype=complex)


#************TOTAL INVERSION PROCEDURE*********#

# We used to cost_func.data_vec cause this is were we store the original good 
# Solution. If we used X_eta we will have the solution computed at the last
# SPSA iteration
Y_inv=scipy.sparse.linalg.lsqr(SPM_obj.B_eta,SPM_obj.A_eta*Cost_func.data_vec)
Y_inv=Y_inv[0] # So that I do not save the additional values it gives me

# We have to modify it a little to get something that can fit in the cost
# Function
Cost_func_inv=Cost_func(np.hstack([Y_inv.real,Y_inv.imag])) 
# To know if the total inversion worked well

#***********Plotting***************************#
# Comparing solution
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

ax.plot(Y.real,marker='*',label='Real part,SPSA')
ax.plot(Y.imag,marker='o',label='Imaginary part,SPSA')

ax.plot(Y_inv.real,marker='p',label='Real part,Total Inverse')
ax.plot(Y_inv.imag,marker='o',label='Imaginary part,Total Inverse')

ax.plot(SPM_obj.Y_eta.real,label='Real part, original data')
ax.plot(SPM_obj.Y_eta.imag,label='Imaginary part, original data')

ax.legend(loc='best')
ax.set_xlabel('# Of component of the bc vector')
ax.set_ylabel('Value (arbitrary unit)')

fig.show()

# Efficiency of the cost function
fig_J=plt.figure()
ax_J=fig_J.add_subplot(1,1,1)

ax_J.plot(Test_SPSA.J,label='SPSA (standard)')
ax_J.plot(Cost_func_inv*np.ones(np.size(Test_SPSA.J)),label='Total Inverse')

ax_J.legend(loc='best')
ax_J.set_yscale('log')

ax_J.set_xlabel('Iteration')
ax_J.set_ylabel('Cost function')

fig_J.show()
