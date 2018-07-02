#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:09:32 2018

@author: koenig.g
"""

#########################################
# Reduced version of Serpent de Mer only#
# For the overelevation, since the rest #
# Does not work up to now. By G.Koenig  #
# The 26/06/2018                        #
# Modified the 26/06/2018 to keep only  #
# Useful functions                      #
#########################################

#***********Packages Import***********#
import numpy as np
#import bohrium as np
import matplotlib.pyplot as plt
import netCDF4
import scipy

#***********Class defining************#

class SerpentdeMer_eta:

    
    #************************************************#
    #              INITIALIZING PART                 #
    #************************************************#
    def __init__(self):
        """Initiliazing the Serpent de Mer class. We will create a grid
        on which we will solve linearized version of Navier-Stokes 
        depth-averaged equation, as given by Devenon (1989). We will make it 
        possible to load data for boundary conditions
        from an external file or to generate them from given values.
        There may be subtleties associated to the matrix forms. Those are 
        better explained in the associated functions.
        
        VARIABLES:
        SELF:
        h : Static water depth, in meters
        eta : Fluctuating water depth (overelevation),in meters
        ubar,vbar: Zonal and meridional depth-averaged velocity, in m.s-1 
        (complex valued)
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
        X_pos,Y_pos : Zonal and Meridional position of the center of a grid 
        cell
        
        grid_size : Size of the grid, in zonalxmeridional form
        dom_size : Size of the domain over which we solve. It is the grid
        less so the boundary points
        
        X=The solution vector in which we put the velocities
        Y= The boundary condition vector, for velocities as well
        A= The solving matrix for the domain of resolution
        B= The projecting matrix that sends boundary conditions into the domain
        """
     
        # INPUT DATA
        self.h=[]
        self.eta=[[],[]]
        self.dx=[]
        self.dy=[]
        self.f=[]
        
        # Size of grid and  others
        self.grid_size=[[],[]]
        self.dom_size=[[],[]]
        
        #Linear algebra object
        self.X=[]
        self.Y=[]
        self.A=[[],[]]
        self.B=[[],[]]
        
        # Grid
        self.X_pos=[[],[]]
        self.Y_pos=[[],[]]
    #************************************************#
    #         GRID IMPORTING/CREATING PART           #
    #************************************************#     

    def set_grid_manually(self,z_size,m_size,f=10e-4,ubar=1.+1.j,
                          vbar=1.+1.j, h=10.,dx=20.,dy=20.,eta=0.2):
        """Generate manually a grid with boundary conditions. Default values 
        are given for all variables except size,longitude and latitude which 
        are a little tricky.
        
        VARIABLES:
        SELF:
        h : Static water depth, in meters
        ubar,vbar: Zonal and meridional depth-averaged velocity, in m.s-1 
        (complex valued)
        eta : Fluctuating water depth (overelevation),in meters
        lon,lat : lon and lat position of a grid cell
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
        grid_size : Size of the data grid, in number of points
        dom_size : Size of the resolution grid, which is the data grid less
        the boundary points, in number of points
        
        NON-SELF:
        z_size,m-size : Zonal and meridional siz of the grid
        lon,lat: Longitude and Latitude of a grid point"""
        
        self.grid_size=[z_size,m_size] # Set the size of the grid
        self.dom_size=[z_size-2,m_size-2] # Set the size of the resolution
        # Domain
        
        self.h=h # The depth
        self.f=f # Coriolis
        self.dx=dx # X-direction size
        self.dy=dy # X-direction size
        
        
        # For ubar,vbar and eta, things are a little different
        self.ubar=np.zeros(self.grid_size,dtype=complex)# Defining it
        
        self.ubar[0,:]=ubar # Limit conditions
        self.ubar[-1,:]=ubar
        self.ubar[:,0]=ubar
        self.ubar[:,-1]=ubar
        
        self.vbar=np.zeros(self.grid_size,dtype=complex)# Defining it
    
        self.vbar[0,:]=vbar # Limit conditions
        self.vbar[-1,:]=vbar
        self.vbar[:,0]=vbar
        self.vbar[:,-1]=vbar
        
        
        self.eta=np.zeros(self.grid_size,dtype=complex) # Overelevation
        self.eta[0,:]=eta # Limit conditions
        self.eta[-1,:]=eta
        self.eta[:,0]=eta
        self.eta[:,-1]=eta
        # Create a grid for positions
        
        self.X_pos=np.array([[self.dx/2. + self.dx*i 
            for j in range(self.grid_size[1])] 
            for i in range(self.grid_size[0])])
        self.Y_pos=np.array([[self.dy/2. + self.dy*j
            for j in range(self.grid_size[1])] 
            for i in range(self.grid_size[0])])
    
          
    #************************************************#
    #               EXAMPLE OF SOLUTIONS             #
    #************************************************#

    def sol_Kelvin_wave(self,sigma=2.31484e-5,g=9.81,u_0=1.+0.5j) :
        """ Here we compute the analytical solution of Kelvin waves
        for the conditions given in the BC.
        VARIABLES:
          
        SELF:
        h : Static water depth, in meters
        ubar,vbar: Zonal and meridional depth-averaged velocity, in m.s-1 
         (complex valued)
        eta : Local overelevation of water level, in m-1
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
        grid_size : Size of the data grid, in number of points
        X_pos,Y_pos : gridded positions of the cells 's centers, in m
          
        NON-SELF:
        sigma : Frequency of the given tide in s-1
        g : Gravity acceleration, in m.s-2
        u_0 : Initial speed of the wave, in m.s-1 (Complex valued)"""
          
        # First we set some useful values
        c=np.sqrt(g*self.h)
        omega=sigma*2.*np.pi # We get in rad.s-1
        k=omega/c # The wavevector
        
        # To be sure
        self.vbar[:,:]=0.
        
        # And still to get an idea
        self.ubar[:,:]=u_0*np.exp(1.j*k*self.X_pos[:,:])\
            *np.exp(-self.f*self.Y_pos[:,:]/c)
        self.eta[:,:]=-1j*sigma/c*self.ubar[:,:]
        
    #***************************************************#
    #                SOLVING METHODS                    #
    #***************************************************#
    
    def solve_eta_jacobi(self,eps,g=9.81,sigma=2.31484e-5):
        """Solve the Helmoltz problem -eta_tt - f**2 eta = -c**2 delta eta 
        with given boundary conditions and for eta=K(x,y)e-i*sigma*t. We use a 
        jacobi method. The equation is linearized and wave speed only depend on
        static water height.
            
        INPUTS:
        SELF:
        h : Static water depth, in meters
        eta : Fluctuating water depth (overelevation),in meters
        dx,dy : Size of a grid cell in x and y directions, in meters
        f : Coriolis frequency, in s-1
            
        NON-SELF: 
        g : Gravity acceleration, in m.s-2
        sigma : Frequency of the given tide in s-1
        omega : Relaxation parameter, between 1.2 and 1.4 for the given problem
        eps : Error tolerancy to stop the computation
            
        OUTPUTS :
        SELF:
        eta : Described earlier"""
            
        # Helping parameters
        c_2=g*self.h 
        # The square of the local speed, to be used in further calculations
    
        beta=-c_2/(sigma**2 - self.f**2 -2.*c_2\
                   *(1./self.dx**2 + 1./self.dy**2))
        
        print('self.dx',self.dx,'self.dy',self.dy)
        # The corrective factor to apply to the laplacian operator
            
        # Error parameters
        err=1. # My initial error
        i=0 # A counter to get the necessary number of iterations
            
        while(err>eps):
            i+=1
            eta_cp=self.eta.copy()
            #Solving using a five points stencil 
            self.eta[1:-1,1:-1]=beta*((self.eta[1:-1,2:]\
               +self.eta[1:-1,:-2])/self.dy**2 + \
                (self.eta[2:,1:-1]+self.eta[:-2,1:-1])/\
                self.dx**2)
            
            #Error computing
            err=scipy.linalg.norm(eta_cp-self.eta)/scipy.linalg.norm(eta_cp)
             # Do not depend on the number of cells.
            # Error computing based on difference between two iterations
         
        print('Converged in '+str(i)+' iterations')
        print('Total error',np.mean(abs(eta_cp-self.eta))*self.grid_size[0]**2)
        print('Error',err)
    #**************************************************#
    #                PLOTTING PART                     #
    #**************************************************#
    
    
    def plot_overelevation(self,angle=True,magnitude=True,nbr_levels=10.) :
        """ A simple routine to do a contourfilled plot of overelevation. But 
        it will be easier than rewriting it everytime.
                
        INPUTS:
        SELF : 
                
        eta : Variable water depth, in meters
        angle : Boolean flag to decide whether plotting the angle map
        magnitude : Boolean flag to decide whether plotting the magnitude map
        levels=number of levels, real number
                
        OUTPUTS : 
        A nice contourfilled plot """
                
        if angle:
            fig=plt.figure(figsize=(12,8))
            ax=fig.add_subplot(1,1,1)
            ax.set_xlim([self.X_pos[0,0],self.X_pos[-1,-1]])
            ax.set_ylim([self.Y_pos[0,0],self.Y_pos[-1,-1]])
            #****First plotting the elevation*********#
            phi_min=np.angle(self.eta).flatten().min()
            phi_max=np.angle(self.eta).flatten().max()
            levels=np.arange(phi_min,phi_max,(phi_max-phi_min)/nbr_levels)
            cax=ax.contourf(self.X_pos,self.Y_pos,np.angle(self.eta),
                            levels=levels,cmap='jet')
                   
            # Colorbar
            cbar=fig.colorbar(cax,orientation='horizontal')
            cbar.set_label('Angle of overelevation (degres)',fontsize=25)
                   
            #***Making fancy plot****************#
            ax.set_xlabel('X distance (meters)',fontsize=30)
            ax.set_ylabel('Y distance (meters)',fontsize=30)
    
            ax.tick_params(labelsize=20)
            fig.show()

        if magnitude :
            fig_1=plt.figure(figsize=(12,8))
            ax_1=fig_1.add_subplot(1,1,1)
            ax_1.set_xlim([self.X_pos[0,0],self.X_pos[-1,-1]])
            ax_1.set_ylim([self.Y_pos[0,0],self.Y_pos[-1,-1]])
            #****First plotting the elevation*********#
            amp_min=abs(self.eta).min()
            amp_max=abs(self.eta).max()
            levels=np.arange(amp_min,amp_max,(amp_max-amp_min)/nbr_levels)
            cax_1=ax_1.contourf(self.X_pos,self.Y_pos,abs(self.eta),
                            levels=levels,cmap='jet')
                   
            # Colorbar
            cbar_1=fig_1.colorbar(cax_1,orientation='horizontal')
            cbar_1.set_label('Amplitude of overelevation (m)',fontsize=25)
                   
            #***Making fancy plot****************#
            ax_1.set_xlabel('X distance (meters)',fontsize=30)
            ax_1.set_ylabel('Y distance (meters)',fontsize=30)
    
            ax_1.tick_params(labelsize=20)
            fig_1.show()
