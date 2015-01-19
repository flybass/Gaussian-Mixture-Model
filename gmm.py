
# coding: utf-8
import numpy as np
import random
from scipy.stats import multivariate_normal

class gmm:
    #set n_comps
    def __init__(self, n_comps=4, delta = 10**-5):
        #call this k
        self.n_comps = n_comps
        self.delta = delta
    
    #data is a matrix n*p (n rows, p dimensional)
    def fit(self, data):
        #starting parameter values
        N,p = data.shape
        k =self.n_comps
        theta_last = self.init_params(data)
        converged = False
        likelihood =0
        while not(converged):
            #1. Compute membership weights
            #this is p(y_i = k | x_i, theta)
            #w is n * k
            w = self.mem_weights(data, theta_last)
            #compute N_k
            N_k = np.sum(w, axis=0)
            #compute the new alphas
            alpha_new = N_k * (1./N)
            #compute the new means (k*p)
            u = np.dot(np.diag(1./N_k),np.dot(w.T, data))
            #compute the new sigmas
            sigmas = self.sig_update(w,N_k, data, u)
            #compute new theta
            new_theta = [(u[t,:],sigmas[t],alpha_new[t]) for t in range(k)]
            converged,likelihood = self.converged(new_theta,data,likelihood)
            theta_last = np.copy(new_theta)
        self.th = theta_last
        self.likelihood = likelihood
        return theta_last
    
    def classify(self, data):
        theta = self.th
        #w is n * k
        w = self.mem_weights(data, theta)
        return [np.argmax(w[i,:]) for i in range(w.shape[0])]

    def mem_weights(self, data, theta):
        #define a function for multivariate guassian probability
        #yields p(x|\mu, \Sigma,alpha)
        def p_x(x,theta_i):
            mu,Sigma, alpha = theta_i
            p = multivariate_normal.pdf(x, mean=mu, cov=Sigma)
            return alpha*p
        #w_ik =p(y_i =k|x_i,Θ)
        #p(y_i =k|x_i,Θ) = \propto α_y p_y (x_i |θ_y)
        N,p = data.shape
        k = self.n_comps
        w = np.empty([N,k])
        for i in range(N):
            for j in range(k):
                w[i,j] = np.copy(p_x(data[i,:], theta[j]))
            if np.sum(w[i,:])!= 0:
                w[i,:]= w[i,:]*1./np.sum(w[i,:])
            else:
                #correcting for numerical instability
                w[i,:]= np.array([1./k]*k)
        return w
        
    def sig_update(self, w,N_k, data,u):
        new_sigs = []
        k = self.n_comps
        N,p = data.shape
        for j in range(k):
            sig = np.zeros([p,p])
            for i in range(N):
                diff = data[i,:] - u[j,:]
                sig += np.copy(w[i,j]*np.outer(diff,diff))
            new_sigs.append((1./N_k[j])*np.copy(sig))
        return new_sigs
        
    def converged(self,theta,data,likelihood):
        def p_x(x,theta_i):
            mu,Sigma, alpha = theta_i
            p = multivariate_normal.pdf(x, mean=mu, cov=Sigma)
            return alpha*p
        new_likelihood = 0
        for i in range(data.shape[0]):
            term_sum = 0
            for j in range(self.n_comps):
                term_sum += p_x(data[i,:],theta[j])
            new_likelihood += np.log(term_sum)
        return abs(likelihood - new_likelihood) < self.delta, new_likelihood
        
        
    def init_params(self,data):
        #pick n_comps random points to be centers
        #make alphas uniform
        #initialize covariances as cov of data
        cov = np.cov(data.T)
        k = self.n_comps
        means = [data[ind,:] for ind in random.sample(xrange(data.shape[0]), k)]
        #and make a tuple to represent each gaussian
        #theta is the list of these tuples
        #each item is (mu, \Sigma, alpha)
        theta = [(u,np.copy(cov), 1.0/k) for u in means]
        #return the intialization theta
        return theta
        
        