#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:27:55 2018

@author: koenig.g
"""

#****************************************#
# Objects and codes for the SPSA method  #
# First in a vanilla scheme. However we  #
# want to quickly include momentum and   #
# Other terms                            #
# Modified the 28/06/2018 for adding ani #
# mations                                #
# Modified the 29/06/2018 for adding a   #
# Mean SPSA computation                  #
# Modified the 02/07/2018 to add a moment#
# -um term                               #
# Modified the 02/07/2018 to add an adap-#
# tative step size                       #
#****************************************#

#*******Packages Import******************#
import numpy as np
import random
import matplotlib.pyplot as plt

class SPSA():
    
    def __init__(self,Y,cost_func,tol,n_iter,gamma,alpha,A,comp=False,
                 anim=False,n_anim=100):
        """ Declaration of the main variables.
        
        INPUTS :
        Y : vector of parameters of the model used, of size n
        cost_func : Cost function used for the optimization
        tol : Tolerance for the error of the cost function
        n_iter :  Maximum number of iteration for the gradient estimate
        gamma,alpha,A : Tuning parameters for the SPSA
        comp: Flag to determine if the parameter vectors will contain complex
        numbers"""
        
        self.k_grad=0 # Number of estimates of the gradient
        self.k_costfun=0 # Number of calls for the model/cost_function
        
        self.Y=np.array(Y,dtype=complex) # Vector of parameters
        self.vec_size=Y.size
        self.comp=comp
        self.c=np.ones(self.vec_size,dtype=complex) # Update for parameters
        self.a=np.eye(self.vec_size,dtype=complex) # weight of the gradient 
        
        self.gamma=gamma
        self.alpha=alpha
        self.A=A # Tuning parameters for the SPSA
        
        self.tol=tol # Tolerance for the error
        self.n_iter=n_iter # Number max of iterations
        
        self.cost_func=cost_func # Importing the cost function
        self.J=[] # An empty list to store the cost function of different
        # iterations
        
        self.fig=[]
        self.ax=None # Used for plotting
        
        self.anim=anim # To determine if we use the animation
        self.n_anim=100 # To determine the frequency of refreshing of the
        # Animation
    #################################################################
    #           FUNCTIONS USED FOR THE SPSA                         #
    #################################################################
        
    def cost_func_wrapper(self,Y):
        """ We use this to increment our counter of call to the cost 
        function"""
 
        self.k_costfun+=1 # We increment
        
        return self.cost_func(Y)
        
    def perturb_vector(self):
        """ We use this function to perturb vector in a symmetric fashion"""
        
        delta=(np.array([random.randint(0,1) for p in \
                         range(self.vec_size)],dtype=complex)-0.5)*2. 
    # Perturbation vector
    
        if self.comp :
            delta+=(np.array([random.randint(0,1) for p in \
                         range(self.vec_size)])-0.5)*2j # Imaginary part
             
        # We return the symetrically perturbed vector and the perturbation
        return self.Y+self.c*delta,self.Y-self.c*delta,self.c*delta
        
    def update_c(self):
        """ Very simple update of c. We do not use ck or anything in a first
        formulation"""

        k=float(self.k_grad) # Just to make the function easier to read        
        self.c=self.c*(k/(k+1))**self.gamma # We multiply by k to remove the 
        #last iteration effect
        
        
    def update_a(self):
        """ Very simple update of a. We do not use ak or anything in a first
        formulation """

        k=float(self.k_grad)        
        
        self.a=self.a*((k+self.A)/(k+1+self.A))**self.alpha
        
    def grad(self):
        
        """Very simple function to estimate the gradient"""
        
        Y_plus,Y_minus,delta_k=self.perturb_vector() # Getting the perturbed
        # vectors
        grad_estim=(self.cost_func_wrapper(Y_plus)-\
        self.cost_func_wrapper(Y_minus))/delta_k # Estimation of the gradient
        
        
        return grad_estim
    
    ##############################################################
    #                 ANIMATION SCHEME                           #
    ##############################################################
    
    def Anim_func(self):
        """Animation to get the evolution of the parameters space"""
        
        if self.ax: # If there already is an object we empty it
             self.ax.clear()
        else : # If not we declare the necessary objects
                self.fig=plt.figure()
                self.ax = self.fig.add_subplot(1,1,1)

        #Normal plotting stuff
        self.ax.plot(self.Y,marker='*',label='Parameters vector')
        self.ax.legend(loc='best')
        self.ax.set_xlabel('# Of component of the bc vector')
        self.ax.set_ylabel('Value (arbitrary unit)')

        self.ax.set_title('Iteration number'+str(self.k_grad))
        plt.pause(.50)
        
        
        
    ##############################################################
    #                    COMPLETE EXAMPLE                        #
    ##############################################################
        
    def SPSA_vanilla(self):
        """ The most basic version of SPSA estimation. """
        
        self.J.append(self.cost_func_wrapper(self.Y)) 
        # We start our first estimate
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            
            self.update_c()
            self.update_a() # We update our gain variables
            

            self.Y-= np.dot(self.a,self.grad()) # We call directly the 
            # gradient estimate in the dot product
            
            # Plotting the animation
            if self.k_grad%self.n_anim==0 : 
                if self.anim :
                    self.Anim_func() # Call the function for animation
            
            self.J.append(self.cost_func_wrapper(self.Y))
            # We append the last value
            
    def SPSA_mean(self,n_mean):
        """ A version of SPSA estimation where you take n_estimation of
        SPSA and take the mean of them to get your gradient estimation.
        
        INPUTS :
        n_mean : Number of iterations in your averaging"""
        
        self.J.append(self.cost_func_wrapper(self.Y)) 
        # We start our first estimate
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            
            self.update_c()
            self.update_a() # We update our gain variables
            
            # Taking the mean of n_estimates
            for i in range(n_mean):
                if i==0:
                    grad_estim=self.grad()
                else :
                    grad_estim+=self.grad()
                    
            grad_estim/=n_mean
            
            # Updating the vector of parameters
            self.Y-= np.dot(self.a,grad_estim)
            
            # Plotting the animation
            if self.k_grad%self.n_anim==0 : 
                if self.anim :
                    self.Anim_func() # Call the function for animation
            
            self.J.append(self.cost_func_wrapper(self.Y))
            # We append the last value
            
    def SPSA_momentum(self,mom_coeff=0.9):
        """ A version of SPSA estimation where you try to consider the last
        estimation of the gradient in your new estimation. For that at each
        time step gradient is estimated by : grad=(A*grad_old + B*grad_new)/
        (A+B). Here A and B are positive coefficients, grad_old is the last
        estimation of the gradient, grad_new is the estimation given by
        perturbations of the parameters's vector.
        
        INPUTS:
            mom_coeff : float, weight of the estimation of the gradient at time
            k in the gradient for time k+1 (A in the formula above)"""
        
        self.J.append(self.cost_func_wrapper(self.Y)) 
        # We start our first estimate
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            
            self.update_c()
            self.update_a() # We update our gain variables
            
            # Gradient estimation
            
            if self.k_grad==1:
                # First estimation of gradient. No momentum term here
                grad_k=self.grad()
                print('Ach,Ich bin in SPSA mit momentum')
            else :
                # Or here we take a momentum formulation
                grad_k=(self.grad()+mom_coeff*grad_k)/(1.+mom_coeff)
            
            
            self.Y-= np.dot(self.a,grad_k) # We call directly the 
            # gradient estimate in the dot product
            
            # Plotting the animation
            if self.k_grad%self.n_anim==0 : 
                if self.anim :
                    self.Anim_func() # Call the function for animation
            
            self.J.append(self.cost_func_wrapper(self.Y))
            # We append the last value
            
    def SPSA_adaptative_step_size(self,mult_size):
        """ A version of SPSA estimation where we compare the angle between
        two evaluation of the gradient to get the step size of the next modi-
        fication of the parameters vector. We use a normal gain sequence ak
        that we multiply by a factor mult_size^cos(theta), where theta designate 
        the angle between the estimation of the gradient at the last step
        and this step estimation of the gradient.
        
        INPUTS :
         mult_size : basis for the multiplicative factor mult_size^cos(theta),
         positive float
            """
        
        self.J.append(self.cost_func_wrapper(self.Y)) 
        # We start our first estimate
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            
            self.update_c()
            self.update_a() # We update our gain variables
            
            # Gradient estimation
            grad_k=self.grad()
            
            # Adapting step size
            if self.k_grad==1:
               # Then we set the factor to one
               theta=1.
               print('Ich bin in Adaptatiev-SPSA !')
            else :
                # And here we try and compute it
                angle=np.dot(grad_k,grad_k_1)/(\
                np.sqrt(np.dot(grad_k,grad_k)*np.dot(grad_k_1,grad_k_1)) \
                +np.finfo(np.float).eps) # I regularize with a machine epsilon
                theta=mult_size**(angle)

            self.Y-= np.dot(self.a,grad_k)*theta # We call directly the 
            # gradient estimate in the dot product
            
            # Saving the gradient value
            grad_k_1=grad_k.copy()
            
            # Plotting the animation
            if self.k_grad%self.n_anim==0 : 
                if self.anim :
                    self.Anim_func() # Call the function for animation
            
            self.J.append(self.cost_func_wrapper(self.Y))
            # We append the last value