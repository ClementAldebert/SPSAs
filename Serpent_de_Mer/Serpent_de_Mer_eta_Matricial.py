#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:15:10 2018

@author: koenig.g
"""

#*****************************************# 
# Matricial methods for the Serpent de Mer#
# Eta. By G.Koenig,the 25/06/2018         #
# Modified the 26/06/2018 to keep only use#
# ful functions                           #
###########################################

#***********Packages Import***********#
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import scipy.sparse as sparse
import scipy.sparse.linalg 
from Serpent_de_Mer_eta import * 

#***********Class defining************#

class SerpentdeMerMatricial_eta(SerpentdeMer_eta):
    
 ##########################################################################
# Solving part for eta                                                   #
##########################################################################

    def create_solution_vector_overelevation(self) :
        """ A function that uses the overelevation matrix to create
        the solution vector used in the linear formulation of the problem.
        
        INPUTS :
        SELF:
        
        eta : overelevation, in m
        dom_size : Size of the resolution grid, which is the data grid less
        the boundary points, in number of points
        
        OUTPUTS :
        X_eta= the vector of overelevation, of shape (grid_size[0]-2,
        grid_size[1]-2)  """
        
        # First we create an empty vector of the appropriate size
        self.X_eta=np.zeros(self.dom_size[0]*self.dom_size[1])
        # Now we can put our values in it and intricate them
        
        self.X_eta=self.eta[1:-1,1:-1].flatten()
        
        
        
    def create_bc_vector_overelevation(self) :
        """ A function to create the boundary condition vector used in the 
        linear formulation of the problem
        
        INPUTS :
        SELF :
        
        eta : overelevation, in m
        grid_size : Size of the data grid, in number of points
        dom_size : Size of the resolution grid, which is the data grid less
        the boundary points, in number of points
        
        OUTPUTS : 
        Y_eta : Boundary condition vector of overelevation, 
        of shape 2*(grid_size[0]+grid_size[1]-2)"""
        
        # First line
        self.Y_eta=np.zeros(self.grid_size[1],dtype=complex) # Declaring it
        self.Y_eta=self.eta[0,:]
        
        # Then we add the sides 
        Y_=np.zeros(2*(self.dom_size[0]),dtype=complex)
        Y_[::2]=self.eta[1:-1,0]
        Y_[1::2]=self.eta[1:-1,-1]
        self.Y_eta=np.hstack([self.Y_eta,Y_])

        # And finally the top
        Y_=np.zeros(self.grid_size[1],dtype=complex)
        Y_=self.eta[-1,:]
        self.Y_eta=np.hstack([self.Y_eta,Y_])
    
    def create_solving_matrix_overelevation(self,g=9.81,
                                            sigma=2.31484e-5) :
        """ A function to create the solving matrix for the linear formulation 
        of the problem.  Due to the important matrix size we use a scipy.sparse 
        package that can be used to handle large vectors and matrices. We use 
        lil_matrix because it is the only sparse format that allows us to set 
        easily values to components. It may however not be optimal for 
        computational purposes.
        
        INPUTS :
        SELF :
        
        h : Static water depth, in meters
        eta : Fluctuating water depth (overelevation),in meters
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
        grid_size : size of the grid
        
        NON-SELF :
        
        g : Gravity acceleration, in m.s-2
        sigma : Frequency of the given tide in s-1
        
        OUTPUTS :
        
        A_eta : The solving matrix for solving the problem"""
       
        # We create simple parameters for the solving
       
        c_2=g*self.h
        # The shape of matrices we want
        shape=(self.dom_size[0]*self.dom_size[1],
            self.dom_size[0]*self.dom_size[1]) # This is gonna give
        # us the size of all the submatrices
       
        # The first terms for the diagonal matrix, they will get flattened
        V_1=c_2/(self.dx**2)*np.ones(self.dom_size)

        # The other terms for the diagonal matrix, they will get flattened
        V_2=c_2/(self.dx**2)*np.ones(self.dom_size)
        
        # Now we're gonna clean V1 and V2 on the sides to ensure we have
        # Something correct
        
        V_1[:,0]=0. # Left hand side
        V_2[:,-1]=0. # Right hand side
       
        # And then we use them to create a 2-diagonal matrix
        # We have to cut slightly the arrays
        diag_vec=[V_1.flatten()[1:],V_2.flatten()[:-1]]
        # Vectors to be put on the diagonals        
        diag_pos=[-1,1]
        # Their positions
        self.A_eta=sparse.diags(diag_vec,diag_pos,shape=shape,dtype=complex)
               
        # The first terms for the diagonal matrix, they will get flattened
        V_1=c_2/(self.dy**2)*np.ones(self.dom_size)

        # The other terms for the diagonal matrix, they will get flattened
        V_2=c_2/(self.dy**2)*np.ones(self.dom_size)      
       
        # Then we use them for a 2 diagonal matrix
        diag_vec=[V_1.flatten()[self.dom_size[1]:],
                  V_2.flatten()[:-self.dom_size[1]]]
        # Vectors to be put on the diagonals        
        diag_pos=[-self.dom_size[1],+self.dom_size[1]]
        # Their positions
        self.A_eta+=sparse.diags(diag_vec,diag_pos,shape=shape)
        
        # Plus I have to add the eigenvalues vector
        diag_vec=sigma**2-self.f**2 -2.*c_2*(1./self.dx**2+1./self.dy**2)
        diag_vec*=np.ones(self.dom_size)
        self.A_eta+=sparse.diags(diag_vec.flatten(),shape=shape)

    def create_projection_matrix_overelevation(self,g=9.81,sigma=2.31484e-5) :
        """ A function to create the projection matrix that uses the boundary 
        conditionsto determine the values of the first points of the grid. We 
        will do the same thing as above and declare 4 submatrices that we will 
        eventually merge.
        
        INPUTS :
        SELF :
        
        h : Static water depth, in meters
        eta : Fluctuating water depth (overelevation),in meters
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
        grid_size : size of the grid
        dom_size : Size of the resolution grid, which is the data grid less
        the boundary points, in number of points
        
        NON-SELF :
        
        g : Gravity acceleration, in2*(self.grid_size[0]+self.grid_size[1]-2) 
        m.s-2
        sigma : Frequency of the given tide in s-1
        
        OUTPUTS :
        
        B_eta : The projection matrix for setting boundary values"""
        
        # We create simple parameters for the solving
       
        c_2=self.h*g
        # We create the sparse matrix that we will fill
        # It must be of size (size of X,size of Y) because it 
        # Sends the boundary conditions of Y in the domain X
        self.B_eta=sparse.lil_matrix((self.dom_size[0]*self.dom_size[1]
        , 2*(self.grid_size[0]+self.grid_size[1]-2)),dtype=complex) 

        # Now it going to get a little simpler. We take care of the first line
        # To avoid for loop, we trick a little
        ind=(np.arange(0,self.dom_size[1]),np.arange(1,self.dom_size[1]+1))
        self.B_eta[ind]=-c_2/self.dy**2
        
        # Now we take care of  THE last line
        ind_1=(self.dom_size[0]-1)*self.dom_size[1]+ \
             np.arange(0,self.dom_size[1])
        ind_2=self.dom_size[1]+2*self.dom_size[0] + \
             3+ np.arange(0,self.dom_size[1])
        ind=(ind_1,ind_2)
        self.B_eta[ind]=-c_2/self.dy**2
        
        # Now we take the left row
        ind_1=self.dom_size[1]*np.arange(0,self.dom_size[0])
        ind_2=self.dom_size[1]+2*np.arange(0,self.dom_size[0])+2
        ind=(ind_1,ind_2)
        self.B_eta[ind]=-c_2/self.dx**2
        
        # And finally, the right row
        ind_1=self.dom_size[1]*(np.arange(0,self.dom_size[0])+1)-1
        ind_2=self.dom_size[1]+2*np.arange(0,self.dom_size[0])+3
        ind=(ind_1,ind_2)
        self.B_eta[ind]=-c_2/self.dx**2

    def reconstruct_overelevation(self) :
        """ A function that reconstructs the overelevation matrix from
        the solution vector.
        
        INPUTS :
        SELF:
        
        X_eta : the vector of overelevation, of shape (grid_size[0]-2,
        grid_size[1]-2)
        dom_size : Size of the resolution grid, which is the data grid less
        the boundary points, in number of points
        
        OUTPUTS :
        eta : The overelevation matrix, in meters. 
        """
        
        #And we resend the values
        self.eta[1:-1,1:-1]=self.X_eta.reshape(self.dom_size)
