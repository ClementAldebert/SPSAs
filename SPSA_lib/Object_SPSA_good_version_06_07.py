#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:51:46 2018

@author: koenig.g
"""

###############################################
# Object of SPSA that store the functions that#
# Have been tested and are functioning so far #
# We're sure that everything is working       #
# Last modification by G.Koenig the 09/07/2018#
# To improve the stochastic direction SPSA    #
###############################################


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
        
        if comp:
            self.Y=np.array(Y,dtype=complex) # Vector of parameters
        else :
            self.Y=np.array(Y) # Vector of parameters
        self.vec_size=Y.size
        self.comp=comp
        if comp :
            self.c=np.ones(self.vec_size,dtype=complex) # Update for parameters
            self.a=np.eye(self.vec_size,dtype=complex) # weight of the gradient 
        
        else :
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
    #################################################################
    #           FUNCTIONS USED FOR THE SPSA                         #
    #################################################################
        
    def cost_func_wrapper(self,Y):
        """ We use this to increment our counter of call to the cost 
        function"""
 
        self.k_costfun+=1 # We increment
        
        return self.cost_func(Y)
    
    #*****PARAMETERS UPDATE***************#
        
    def perturb_vector(self):
        """ We use this function to perturb vector in a symmetric fashion"""
        
        delta=(np.array([random.randint(0,1) for p in \
                         range(self.vec_size)])-0.5)*2. 
    # Perturbation vector
    
        if self.comp :
            delta=(np.array([random.randint(0,1) for p in \
                         range(self.vec_size)],dtype=complex)-0.5)*2.
    
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
            # Modifying them to get the direction, but we cannot
            # Use simply the above function
            Y_plus=self.Y+self.dir_mat*delta_k
            Y_minus=self.Y-self.dir_mat*delta_k
        
        
        grad_estim=(self.cost_func_wrapper(Y_plus)-\
        self.cost_func_wrapper(Y_minus))/(2.*delta_k)
                    # Estimation of the gradient
        
        if self.dir_mat is not None:
            # Correction for the directions that we want
            grad_estim=self.dir_mat*grad_estim # We take advantage 
        
        return grad_estim
    
    def grad_one_sided(self):
        
        """Very simple function to estimate the gradient in a one_sided fashion
         """
        
        Y_plus,Y_minus,delta_k=self.perturb_vector() # Getting the perturbed
        # vectors
        
        if self.dir_mat: # If A is not null
            # Modifying them to get the direction
            Y_plus=self.dir_mat*Y_plus
        
        
        grad_estim=(self.cost_func_wrapper(Y_plus)-self.J[-1])/delta_k 
        # Estimation of the gradient
        
        if self.dir_mat is not None:
            # Correction for the directions that we want
            grad_estim=self.dir_mat*grad_estim
        
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
        
        # Saying a word 
        print('Ach, ich bin in SPSA vanilla')
        
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
            
            self.Y-=np.dot(self.a,self.grad_one_sided()) #  We call directly
            # the gradient estimate
            
            # Plotting the animation
            if self.k_grad%self.n_anim==0: # Frequency of refreshing
                if self.anim :
                    self.Anim_func() # Refreshing the plot
                    
            self.J.append(self.cost_func_wrapper(self.Y)) #Appending the cost
            # Function
            
            
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
                
                self.Y-=np.dot(self.a,self.grad())*float(self.vec_size)/\
                    float(batch_size)# We call directly the 
                # gradient estimate in the dot product
                
                # Plotting the animation
                if self.k_grad%self.n_anim==0 : 
                    if self.anim :
                        self.Anim_func() # Call the function for animation
                
                self.J.append(self.cost_func_wrapper(self.Y))
                # We append the last value
                
    def SPSA_stochastic_direction_momentum(self,batch_size,mom_coeff=0.9):
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
                    grad_k=self.grad()
                    print('Ach, ich bin in SPSA mit stochastic direction und momentum')
                else :
                    # Or here we take a momentum formulation. However, 
                    # We do not want to artificially get a too low gradient.
                    # So we introduce a loop formulation. I am pretty
                    # Sure that there is a more beautiful way to do this with
                    # Numpy
                    grad_est=self.grad()
                    
                    for i in range(grad_est.size):
                        if abs(grad_est[i])>0.: # If the direction is non-null
                            grad_k[i]=(mom_coeff*grad_k[i]+grad_est[i])/\
                            (1.+mom_coeff)
                        else :
                            grad_k[i]=grad_k[i] # We just keep it
                                            
                
                self.Y-=np.dot(self.a,grad_k)*float(self.vec_size)/\
                    float(batch_size) # We call directly the 
                # gradient estimate in the dot product
                
                # Plotting the animation
                if self.k_grad%self.n_anim==0 : 
                    if self.anim :
                        self.Anim_func() # Call the function for animation
                
                self.J.append(self.cost_func_wrapper(self.Y))
                # We append the last value
                
    def SPSA_stochastic_RMS_prop(self,batch_size,mom_coeff=0.5):
            """ A version of SPSA estimation where we search the gradient on 
            subspace and add it to the previously obtained gradients. The goal 
            is to break the isotropy of SPSA and make it able to solve not so 
            well conditionned problems.
            
            Here we introduce a RMS prop formulation for computing the hessian,
            hoping that this will allow us to adapt the step size automatically
            and solve badly conditionned problems.
        
            INPUTS:
            batch_size : Number of parameters taken for each direction
            mom_coeff : Weight of the last gradient estimate in the averaging 
            process."""
        
            self.J.append(self.cost_func_wrapper(self.Y)) 
            # We start our first estimate
            self.a*=10.
#            self.c*=10.
            
            while self.J[-1] > self.tol and self.k_grad < self.n_iter :
                # We iterate until we reach a certain estimate or a 
                # Certain number of iterations
            
                self.k_grad+=1                    # And here we try and compute it
                
                self.update_c()
                self.update_a() # We update our gain variables
                self.update_dir_mat(batch_size) # We update the stochastic
                # Gradient estimation
                
                if self.k_grad==1:
                    # First estimation of gradient. No momentum term here
                    grad_k=self.grad()
                    print('Ach, ich bin in SPSA mit stochastic direction und RMS prop')
                    v=np.ones(grad_k.size) # We initialize our hessian inverse
                    
                    grad_estim=grad_k.copy()
                else :
                    self.dir_mat*=inv_v
                    # Computing the gradient
                    grad_k=self.grad()
                    # Here we approximate the hessian using a local quadratic
                    # Approximation and a momentum formulation for the hessian
                    v=mom_coeff*v.copy()+(1.-mom_coeff)*(grad_k**2) # We take the 
                    # square power of each component
                    grad_estim=mom_coeff*grad_estim.copy()+(1.-mom_coeff)*grad_k
                
                # However, we need the invert of v in computations. And it gonna
                # filled with 0's. So we have to use some trick
                inv_v=[]
                for n in v:
                    if n > 0.:
                        inv_v.append(np.sqrt(1./n))
                    else :
                        inv_v.append(0.)
                # And now that we have filled it
                self.Y-=np.dot(self.a,grad_estim)*inv_v
                
                # Plotting the animation
                if self.k_grad%self.n_anim==0 : 
                    if self.anim :
                        self.Anim_func() # Call the function for animation
                
                self.J.append(self.cost_func_wrapper(self.Y))
                # We append the last value
