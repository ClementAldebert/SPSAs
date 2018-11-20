#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:28:28 2018
@author: koenig.g
"""

###############################################
# Object of SPSA that store the functions that#
# Have been tested and are functioning so far #
# We're sure that everything is working       #
# Last modification by G.Koenig the 06/07/2018#
# Added some variables to save vectors of par-#
# meters across iterations and introduction of#
# Accelerating SPSA , the 12/07/2018, by C.   #
# Aldebert and G.Koenig                       # 
# Modification of the momentum term with sto- #
# Chastic direction following an article about#
# SGD with Momentum. By G.Koenig, the         #
# 19/07/2018                                  #
# Modified the Momentum SPSA the 20/07/2018   #
# So that the momentum is not normalized any- #
# More normalized                             #
# Modified the 20/08/2018 so that we can have #
# A weak constraint on the cost function to   #
# limit the domain of interest (G.Koenig)     #
# Added a line-search algorithm and a BFGS reg#
# -ression technique (13/11/2018, G.Koenig)   #
###############################################


#*******Packages Import******************#
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg

class SPSA():
    
    def __init__(self,Y,cost_func,tol,n_iter,gamma,alpha,A,
                 anim=False,n_anim=100,save_Y=False,n_Y=100,bc=None):
        """ Declaration of the main variables.
        
        INPUTS :
        Y : vector of parameters of the model used, of size n
        cost_func : Cost function used for the optimization
        tol : Tolerance for the error of the cost function
        n_iter :  Maximum number of iteration for the gradient estimate
        gamma,alpha,A : Tuning parameters for the SPSA
        comp: Flag to determine if the parameter vectors will contain complex
        numbers
        anim,n_anim : Flag to determine if we have an animate plot to project
        the evolution of the parameter vector, and the frequency at which this
        plot is refreshed
        save_Y : Flag to determine saving or no of the parameter vector between
        iterations, and the frequency at which we save it
        bc : Matrix of boundary conditions, must be of shape (2,n)"""
        
        self.k_grad=0 # Number of estimates of the gradient
        self.k_costfun=0 # Number of calls for the model/cost_function
        
        self.Y=np.array(Y) # Vector of parameters
        self.vec_size=Y.size
        
        self.c=np.ones(self.vec_size) # Update for parameters
        self.a=np.eye(self.vec_size) # weight of the gradient 
        
        self.gamma=gamma
        self.alpha=alpha
        self.A=A # Tuning parameters for the SPSA
        
        self.tol=tol # Tolerance for the error
        self.n_iter=n_iter # Number max of iterations
        
        self.cost_func=cost_func # Importing the cost function
        self.J=[] # An empty list to store the cost function of different
        # iterations

        self.dir_mat=None # A stochastic direction matrix
        
        self.fig=[]
        self.ax=None # Used for plotting
        
        self.anim=anim # To determine if we use the animation
        self.n_anim=100 # To determine the frequency of refreshing of the
        # Animation
        
        self.save_Y=save_Y # To determine if we save the vectors of parameters
        self.n_Y=n_Y # The frequency to save this vector
        self.Y_his=self.Y.copy() # An empty list to save the parameters vectors
        # Storing the vector of parameters
        
        self.acc=1  # Acceleration factor (product of thetas in accelerating method)
        self.max_acc=5 #Maximal acceleration factor
        
        self.boundaries=None#np.ones(self.vec_size) # Setting the boundary positions
        self.dom_center=np.zeros(self.vec_size) # Position of the center of
        # The domain
        self.lim_coeff=0. # Coeffcient of increase of the weak constraint
        
        #line-search related variables
        self.grad_his=None # This is used to store the gradient estimations
        self.grad_estim=[] # An empty vector at first
        self.c1=1e-3 # Armijo parameter for line-search. Hard to tune. It is ok to set it to 0

        # BFGS related variables
        self.B=[] # Storing the hessian approximation
        self.y=[] # Storing the hessian approximation for the last step

    #################################################################
    #           FUNCTIONS USED FOR THE SPSA                         #
    #################################################################
        
    def cost_func_wrapper(self,Y):
        """ We use this to increment our counter of call to the cost 
        function. The second part is used to weakly constrain the evaluation of
        cost function in a given zone by setting exponential values that grows
        very quickly as soon as we get out of the desired range. This range
        is given by bnd_pos. The increase is tuned up by lim_coeff"""
 
        self.k_costfun+=1 # We increment
        
        return self.cost_func(Y) #+ self.lim_coeff*(np.exp(Y-self.dom_center)\
                              #/self.boundaries)
    
    #*****PARAMETERS UPDATE***************#
        
    def perturb_vector(self):
        """ We use this function to perturb vector in a symmetric fashion"""
        
        delta=(np.array([random.randint(0,1) for p in \
                         range(self.vec_size)])-0.5)*2. 
    # Perturbation vector
    
             
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
        
    def update_dir_mat(self,batch_size):
        """ Update of the direction matrix.
        
        INPUTS
        batch_size : Number of non zero diagonal component of the matrix"""
        
        #We create a random diagonal matrix of 0 and 1
        # That will determine the component of the vector of parameters
        # That will get updated
                
        self.dir_mat=np.hstack([np.ones(batch_size),\
                                np.zeros(self.Y.size-batch_size)]) 
        # We initialize it
        np.random.shuffle(self.dir_mat) 
        
        
    #*******GRADIENT FORMULATION*********#
        
    def grad(self):
        
        """Very simple function to estimate the gradient"""
        
        Y_plus,Y_minus,delta_k=self.perturb_vector() # Getting the perturbed
        # vectors
        
        if self.dir_mat is not None: # If dir_mat is not null
            # Modifying them to get the direction but we cannot
            # Use simply the above function
            Y_plus=self.Y+self.dir_mat*delta_k
            Y_minus=self.Y-self.dir_mat*delta_k

        grad_estim=(self.cost_func_wrapper(Y_plus)-\
            self.cost_func_wrapper(Y_minus))/(2.*delta_k)
        # Estimation of the gradient
        
        if self.dir_mat is not None:
            # Correction for the directions that we want
            grad_estim=self.dir_mat*grad_estim 
        
        return grad_estim
    
    def grad_one_sided(self):
        
        """Very simple function to estimate the gradient in a one_sided fashion
         """
        
        Y_plus,Y_minus,delta_k=self.perturb_vector() # Getting the perturbed
        # vectors
        
        if self.dir_mat is not None: # If A is not null
            # Modifying them to get the direction
            Y_plus=self.dir_mat*Y_plus
        
        
        grad_estim=(self.cost_func_wrapper(Y_plus)-self.J[-1])/delta_k 
        # Estimation of the gradient
        
        if self.dir_mat is not None:
            # Correction for the directions that we want
            grad_estim=self.dir_mat*grad_estim
        
        return grad_estim


    
     #*****LINE-SEARCH algorithm******#

    def line_search(self,n_try):
       """ A line search algorithm to find the best update parameters
       given a certain estimation of the cost function and of its gradient
       
       INPUTS:
       n_try : Number of try before rejecting step
       
       """
       # We initialize a counter as a security to avoid the worst directions
       count=0
       # We initialize the step at one  and then reduce it to see what happens
       # Here we use a simple gradient descent
       if self.k_grad==1 :
           cost_func_estim=self.cost_func_wrapper(self.Y-self.a@self.grad_estim)
           # Jus to track it
           print('Step size is',self.a)
           print('And the perturbation',self.a@self.grad_estim)
           print('And the cost function of',cost_func_estim)
           print('And the last cost function is', self.J[-1])
           print('And the condition on that cost function is',
               self.J[-1]-self.c1*np.dot(self.a@self.grad_estim,self.a@self.grad_estim))

           while  cost_func_estim > \
               self.J[-1]-self.c1*np.dot(self.a@self.grad_estim,self.a@self.grad_estim) :
               # We simply here enforce armijo condition, no exact line-search
               
               # We add the count and check the safety
               count+=1
               if count>n_try :
                   break
               # Updating the step size
               self.a/=cost_func_estim/max(self.J[-1],(self.J[-1]-self.c1*np.dot(self.a@self.grad_estim,self.a@self.grad_estim)))
               # We divide it by the ratio between the cost function that we
               # Get and the one we expected with linear theory. Here we add a 
               # Safety so that the step size does not become negative
               
               # New cost function estimation
               cost_func_estim=self.cost_func_wrapper(self.Y-self.a@self.grad_estim)
               
               # Jus to track it
               print('Step size is',self.a)
               print('And the perturbation',self.a@self.grad_estim)
               print('And the cost function of',cost_func_estim)
               print('And the last cost function is', self.J[-1])
               print('And the condition on that cost function is',
                   self.J[-1]-self.c1*np.dot(self.a@self.grad_estim,self.a@self.grad_estim))

           # We test the safety to avoid bad timesteps
           if count < n_try+1 :
               # Once it's done we save the data
               self.Y-=self.a@self.grad_estim
               self.J.append(cost_func_estim)
               # Updating the step size
               self.step_size/=cost_func_estim/(self.J[-1]-self.c1*np.dot(self.a@self.grad_estim,self.a@self.grad_estim))
           else :
               self.J.append(self.J[-1])
               # Restarting the step size
               self.a=1e-3*np.eye(self.vec_size)
           # Now we can see for the next step, we can use an exact solving for example
    
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
        
        # Saying a word 
        print('Ach, ich bin in SPSA vanilla')
        
        self.J.append(self.cost_func_wrapper(self.Y)) 
        # We start our first estimate
        print(self.J)
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            
#            self.update_c()
#            self.update_a() # We update our gain variables
            
            if self.boundaries is None :
                self.Y-= np.dot(self.a,self.grad()) # We call directly the 
                # gradient estimate in the dot product
                self.J.append(self.cost_func_wrapper(self.Y))
                # We append the last value
            
            else :
                Y_temp=self.Y-np.dot(self.a,self.grad())
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))
            
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])

            
    def SPSA_one_sided(self):
        """The basic version of SPSA with a one-sided estimation """
        
        print('Ach, ich bin in SPSA one-sided')
        
        self.J.append(self.cost_func_wrapper(self.Y))
        # First estimate of the cost function
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
            
            self.k_grad+=1
            
            self.update_c()
            self.update_a() # Updating the gain variables
            
            if self.boundaries is None :
                self.Y-= np.dot(self.a,self.grad_one_sided())# We call directly the 
                # gradient estimate in the dot product
                self.J.append(self.cost_func_wrapper(self.Y))
                # We append the last value
            
            else :
                Y_temp=self.Y-np.dot(self.a,self.grad_one_sided())
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))
            
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])

            
    def SPSA_mean(self,n_mean):
        """ A version of SPSA estimation where you take n_estimation of
        SPSA and take the mean of them to get your gradient estimation.
        
        INPUTS :
        n_mean : Number of iterations in your averaging"""
        
        print('Ach, ich bin in SPSA mean')
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
            


            if self.boundaries is None :
                self.Y-= np.dot(self.a,grad_estim)
                self.J.append(self.cost_func_wrapper(self.Y))
            
            else :
                Y_temp=self.Y-np.dot(self.a,grad_estim)
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))

            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])
           

            
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
                grad_estim=self.grad()
                print('Ach,Ich bin in SPSA mit momentum')
            else :
                # Or here we take a momentum formulation
                grad_estim=self.grad()+mom_coeff*grad_estim
            
            if self.boundaries is None :
                self.Y-= np.dot(self.a,grad_estim)
                self.J.append(self.cost_func_wrapper(self.Y))
            
            else :
                Y_temp=self.Y-np.dot(self.a,grad_estim)
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))

            
            
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])
            
            
    def SPSA_adaptative_step_size(self,mult_size):
        """ A version of SPSA estimation where we compare the angle between
        two evaluation of the gradient to get the step size of the next modi-
        fication of the parameters vector. We use a normal gain sequence ak
        that we multiply by a factor mult_size^cos(theta), where theta designate 
        the angle between the estimation of the gradient at the last step
        and this step estimation of the gradient.
        Here we do not update the gain a, we adapt the step size only taking
        into account the angle between two gradient estimates.
        
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
            
            # Gradient estimation
            grad_estim=self.grad()
            
            # Adapting step size
            if self.k_grad==1:
               # Then we set the factor to one
               theta=1.
               print('Ich bin in Adaptatiev-SPSA !')
            else :
                # And here we try and compute it
                angle=np.dot(grad_estim,grad_k_1)/(\
                np.sqrt(np.dot(grad_estim,grad_estim)*np.dot(grad_k_1,grad_k_1)) \
                +np.finfo(np.float).eps) # I regularize with a machine epsilon
                theta=mult_size**(angle)

            
            if self.boundaries is None :
                self.Y-= np.dot(self.a,grad_estim)*theta
                self.J.append(self.cost_func_wrapper(self.Y))
                grad_k_1=grad_estim.copy()
            else :
                Y_temp=self.Y-np.dot(self.a,grad_estim)*theta
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                    grad_k_1=np.zeros(self.vec_size)
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))
                    grad_k_1=grad_estim.copy()
            
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])
            
                    
    def SPSA_accelerating_step_size(self,mult_size):
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
            
#            self.update_c()
            
            # Gradient estimation
            grad_estim=self.grad()
            
            # Adapting step size
            if self.k_grad==1:
               # Then we set the factor to one
               theta=1.
               print('Ich bin in Accelerating-SPSA !')
            else :
                # And here we try and compute it
                angle=np.dot(grad_estim,grad_k_1)/(\
                np.sqrt(np.dot(grad_estim,grad_estim)*np.dot(grad_k_1,grad_k_1)) \
                +np.finfo(np.float).eps) # I regularize with a machine epsilon
                theta=mult_size**(angle)
                self.acc*=theta
                if self.acc > self.max_acc: self.acc=self.max_acc
            
            if self.boundaries is None :
                self.Y-= np.dot(self.a,grad_estim)*theta
                self.J.append(self.cost_func_wrapper(self.Y))
                grad_k_1=grad_estim.copy()
            else :
                Y_temp=self.Y-np.dot(self.a,grad_estim)*theta
                
                if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                    # We check wheter any of the value it outside of bounds
                    self.J.append(self.J[-1]) # There is no need to update
                    # The vector of parameters or to compute its cost function 
                    # Again
                else :
                    # If nothing, we proceed normally
                    self.Y=Y_temp.copy()
                    self.J.append(self.cost_func_wrapper(self.Y))
                    grad_k_1=grad_estim.copy()
            
            
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])
            

            
    def SPSA_stochastic_direction(self,batch_size):
            """ A version of SPSA estimation where we search the gradient on 
            subspace and add it to the previously obtained gradients. The goal 
            is to break the isotropy of SPSA and make it able to solve not so 
            well conditioned problems. We tried it with one-sided SPSA however
            it did not seem to work great.
            
            INPUTS:
            batch_size : Number of parameters taken for each direction"""
        
            print('Ach, ich bin in SPSA with stochastic direction')
            self.J.append(self.cost_func_wrapper(self.Y)) 
            # We start our first estimate
            
            while self.J[-1] > self.tol and self.k_grad < self.n_iter :
                # We iterate until we reach a certain estimate or a certain 
                # Number of iterations
                self.k_grad+=1
                
                self.update_c()
                self.update_a() # We update our gain variables
                self.update_dir_mat(batch_size) # We update the stochastic
                # Directions matrix
                
                if self.boundaries is None :
                    self.Y-=np.dot(self.a,self.grad())*float(self.vec_size)/\
                        float(batch_size)
                    self.J.append(self.cost_func_wrapper(self.Y))
            
                else :
                    Y_temp=self.Y-np.dot(self.a,self.grad())*float(self.vec_size)/\
                    float(batch_size)
                
                    if (Y_temp<self.boundaries[0,:]).any() or \
                    (Y_temp>self.boundaries[1,:]).any()  :
                        self.J.append(self.J[-1]) 
                    else :
                    # If nothing, we proceed normally
                        self.Y=Y_temp.copy()
                        self.J.append(self.cost_func_wrapper(self.Y))
            
                # Plotting the animation
                if self.anim :
                    if self.k_grad%self.n_anim==0 : 
                        self.Anim_func() # Call the function for animation

                # Saving the data
                if self.save_Y :
                    if self.k_grad%self.n_Y==0 :
                        self.Y_his=np.vstack([self.Y_his,self.Y])
                
    def SPSA_stochastic_direction_momentum(self,batch_size,mom_coeff=0.8):
            """ A version of SPSA estimation where we search the gradient on 
            subspace and add it to the previously obtained gradients. The goal 
            is to break the isotropy of SPSA and make it able to solve not so 
            well conditionned problems.
            
            Here we introduce a momentum formulation, we hope that the momentum
            averaging will create a vector that approximate better the real
            gradient than the plain SPSA over the number of iterations
        
            INPUTS:
            batch_size : Number of parameters taken for each direction
            mom_coeff : Weight of the last gradient estimate in the averaging 
            process."""
        
            self.J.append(self.cost_func_wrapper(self.Y)) 
            # We start our first estimate
            
            while self.J[-1] > self.tol and self.k_grad < self.n_iter :
                # We iterate until we reach a certain estimate or a 
                # Certain number of iterations
            
                self.k_grad+=1
                
                self.update_c()
                self.update_a() # We update our gain variables
                self.update_dir_mat(batch_size) # We update the stochastic
                # Directions matrix
                # Gradient estimation
                
                if self.k_grad==1:
                    # First estimation of gradient. No momentum term here
                    grad_estim=self.grad()
                    print('Ach, ich bin in SPSA mit stochastic direction und momentum')
                else :
                    # We inspire ourselves of the formulation found on
                    #https://bl.ocks.org/EmilienDupont/f97a3902f4f3a98f350500a3a00371db
                    grad_estim=mom_coeff*grad_estim+self.grad()#/np.sqrt(float(self.k_grad))+self.grad()
                # Getting new value                            


                if self.boundaries is None :
                    self.Y-=np.dot(self.a,grad_estim)*float(self.vec_size)/float(batch_size) 
                    self.J.append(self.cost_func_wrapper(self.Y))
                    grad_k_1=grad_estim.copy()
                else :
                    Y_temp=self.Y-np.dot(self.a,grad_estim)*float(self.vec_size)/float(batch_size) 
                
                    if (Y_temp<self.boundaries[0,:]).any() or \
                        (Y_temp>self.boundaries[1,:]).any()  :
                        self.J.append(self.J[-1])

                    else :
                    # If nothing, we proceed normally
                        self.Y=Y_temp.copy()
                        self.J.append(self.cost_func_wrapper(self.Y))
                        grad_k_1=grad_estim.copy()
                # Plotting the animation
                if self.anim :
                    if self.k_grad%self.n_anim==0 : 
                        self.Anim_func() # Call the function for animation

                # Saving the data
                if self.save_Y :
                    if self.k_grad%self.n_Y==0 :
                        self.Y_his=np.vstack([self.Y_his,self.Y]) 


    def SPSA_BFGS(self,n_restart=5,n_try=4):
        """ A SPSA algorithm with a line search included.  
        It needs to save data at each iteration. It uses a line search and
        a SPSA approximation. It is up to now not compatible with the boundary 
        formulation
        
        INPUTS :
        n_restart : Number of iterations before restarting the BFGS approximation
        n_try : Number of try in the line-search"""
        
        # Saying a word 
        print('Ach, ich bin in SPSA BFGS')
        
        # Saving the first iteration of cost function
        self.J.append(self.cost_func_wrapper(self.Y))
        
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            self.k_grad+=1
            # For the first step we use a simple gradient descent
            if self.k_grad==1:
                self.grad_estim=self.grad() # Estimating the gradient

                self.line_search() # Now we call a line search algorithm
                # We initialize the hessian estimate with an identity matrix
                self.B=np.diag(np.ones(self.grad_estim.size))
                # And here the direction is equal to the gradient multiplied by
                # The step size
                self.grad_his=self.grad_estim
                # Recording the direction
                
            # Then we can use the BFGS for the next step
            else :
                # First we estimate the gradient
                self.grad_estim=self.grad() # Estimating the gradient
                # And use it to estimate the local curvature
                self.y=self.grad_estim-self.grad_his
                # And now we store the gradient estimate in grad_his
                self.grad_his=self.grad_estim

                # Now we can determine the new direction
                # For compatibility with the rest of the code we store it in
                # the self.grad_estim variable
                self.grad_estim=scipy.linalg.solve(self.B,self.grad_estim)
                # Then we prepare the two matrices of update
                # First we use an outer/direct product of the curvature estimate
                # That we normalize by the last step
                u=np.outer(self.y,self.y)/np.dot(self.y,np.dot(self.a*self.grad_estim))
                # Then we use another matrix proportionnal to the product of 
                # The last gradient estimation. 
                # We do not use the direction vector but the direction
                # Vector multiplied by the step size. But since the step size 
                # Is a scalar present on both numerator and denominator, we can
                # cancel it
                v=-np.outer(np.dot(self.B,self.grad_estim),np.dot(self.B,self.grad_estim))/(np.dot(self.grad_estim,np.dot(self.B,self.grad_estim)))
                # We can update b
                self.B+=u+v
                # I am trying a restart just to see what happens
                if self.k_grad%n_restart==0 :
                    self.B=np.diag(np.ones(self.grad_estim.size))
                # Now we can do the line search
                self.line_search(n_try=n_try)

            # Saving the last iteration of the cost function 

            print("Hessian estimation",self.B)
            self.Y_his=np.vstack([self.Y_his,self.Y]) # Saving the data


    def SPSA_smart(self,n_mean,mom_coeff=0):
        """ A SPSA algorithm that approximate the local gradient by a gradient estimation
        alongside the last computed gradient and another one alongside a random
        direction orthogonal to the previous gradient. The algorithm is expected
        to be highly sensitive to the first computed gradient, which is estimated by a
        mean-classical-SPSA algorithm.
        
        /!\ Note: in this algorithm version, I do not update the values of a and c
        for the moment
        
        INPUTS :
        n_mean: number of estimations to get the first gradient approximation
        mom_coeff: coefficient for the momentum"""
        
        # Saying a word 
        print('Hello, I am a smart SPSA (not that smart for the moment ...)')

        # Saving the first iteration of cost function
        self.J.append(self.cost_func_wrapper(self.Y))
            
            
        # loop of smart SPSA
        while self.J[-1] > self.tol and self.k_grad < self.n_iter :
            # We iterate until we reach a certain estimate or a certain number
            # Of iterations
        
            if self.k_grad==0:
                # first iteration based on a mean SPSA to get the best gradient estimation
                # as possible to start the algorithm
                # Taking the mean of n_estimates
                for i in range(n_mean):
                    delta_k = self.c * (np.array([random.randint(0,1) for p in range(self.vec_size)])-0.5)*2.
                    Y_plus = self.Y + delta_k
                    Y_minus = self.Y - delta_k
                    if i==0:
                        grad_estim = delta_k * (self.cost_func_wrapper(Y_plus)-\
                                self.cost_func_wrapper(Y_minus))/(2.*self.c**self.vec_size)
                    else :
                        grad_estim += delta_k * (self.cost_func_wrapper(Y_plus)-\
                                self.cost_func_wrapper(Y_minus))/(2.*self.c**self.vec_size)
                
                    grad_estim/=n_mean
            else:
                # estimate alongside the previous gradient (= tangent)
                # note that norm(delta_k) = c^nb_param
                delta_k = self.c * grad_estim / math.sqrt(np.dot(grad_estim,grad_estim))
                Y_plus = self.Y + delta_k
                Y_minus = self.Y - delta_k
                grad_estim_new = delta_k * (self.cost_func_wrapper(Y_plus)-\
                    self.cost_func_wrapper(Y_minus))/(2.*self.c**self.vec_size)
                # estimate alongside a random direction orthogonal to the tangent
                # again, norm(delta_k) = c^nb_param
                # x_i ~ N(0,1), i = 1,..., nb of paramaters
                x = np.array([random.gauss(0,1) for p in range(self.vec_size)])
                # remove the contribution in the direction of the previous
                # direction by Gram-Schmidt decomposition
                x -= np.dot(delta_k,x) * delta_k / np.dot(delta_k,delta_k)
                # rescale to get the desired norm of perturbation
                delta_k2 = self.c * x / math.sqrt(np.dot(x,x))
                # gradient estimation (add to the first one)
                Y_plus = self.Y + delta_k2
                Y_minus = self.Y - delta_k2
                grad_estim_new2 = delta_k2 * (self.cost_func_wrapper(Y_plus)-\
                    self.cost_func_wrapper(Y_minus))/(2.*self.c**self.vec_size)
                grad_estim_new += grad_estim_new2
                # add a momentum effect
                grad_estim = grad_estim * mom_coeff + grad_estim_new
                print(grad_estim_new,grad_estim_new2,grad_estim)

            self.k_grad+=1
            self.Y-= np.dot(self.a,grad_estim)  # update the parameter values
            self.J.append(self.cost_func_wrapper(self.Y))   # new cost function value
         
            # Plotting the animation
            if self.anim :
                if self.k_grad%self.n_anim==0 : 
                    self.Anim_func() # Call the function for animation

            # Saving the data
            if self.save_Y :
                if self.k_grad%self.n_Y==0 :
                    self.Y_his=np.vstack([self.Y_his,self.Y])
           
