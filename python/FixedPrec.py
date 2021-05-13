import sys
import numpy as np
from numpy.linalg import norm

#sys.path.append('./functions')
from util_func_v2 import CG_Steinhaug_matFree, Hessian_times_vec, updateYS, CG_Steinhaug_BFGS_matFree, BFGS_hessian_times_vec, BFGS_updateYS

#function [xvec,fvec,gvec] = TR_fixed(x0,fun,hessian,delta0,eps,budget)
def TR_fixed(x0, fun, hessian, delta0, eps, budget):
    """

    :param x0: initial guess
    :param fun: function handle
    :param hessian: update hessian or not? (I think)
    :param delta0: initial trust region radius
    :param eps: problem tolerance
    :param budget: budget constraint (maybe max number of iterations
    :return: search direction for trust region subproblem
    """

    ## Check for hessian value (seeing if it's 0 or 1, is this just boolean)
    # Not sure what this does. Need Hessian to equal 1 or 0 or else we break
    if not (hessian == 0 or hessian == 1):
        print("hessian value must be either 0 or 1.")
        return


    #X=x0[:]
    n = x0.shape[0]                                                             # dimension of the problem


    ##### INITIALIZATION #####

    #eta_good = 0.01
    #eta_great = 0.1
    #gamma_dec = 0.5
    #gamma_inc = 2
    delta = delta0
    max_delta = 1e1
    memory = 50#min(30, len(x0))
    sr1_tol = 1e-8
    n1 = 0.01                                                                  # n1 denotes eta_1 such that 0 < eta_1 <= eta_2 .< 1
    n2 = 0.5                                                                # n2 denotes eta_2
    gamma1 = 0.5                                                              # 0 < gamma_1 <= gamma_2 .< 1
    gamma2 = 2
    n0 = 0.04*n1                                                              # n0 denotes eta_0 such that eta_0 .< 0.5*(eta_1)
    kg= (0.5*(1-n2)-n0)-1                                                     # kg denotes kappa_g such that kappa_g .< 0.5*(1-eta_2) -eta_0
    epsTR = 1.0e-6

    ## Computing f0 ##

    #xk = x0.reshape((n,1))
    xk = x0



    fvec = []    # might want to change since empty lists, not vectors
    gvec = []
    xvec = []
    #S = np.zeros(0)
    #Y = np.zeros(0)
    S = []
    Y = []
    #B0 = np.eye(n)
    #H = np.zeros((n,n))


    #S = np.zeros((n,1))
    #Y = np.zeros((n,1))
    #H = np.eye(n)  #Diagonal(ones(n,n)) # sets hessian to be indentity

    for k in range(budget):
        if k == 0:
            f,g = fun(xk) # evaluate function and gradient

        # check if gradient is sufficiently small
        if np.linalg.norm(g) <= eps/(1+kg): # or delta < 1e-8:  #if np.linalg.norm(g) <= eps/(1+kg) or delta < 2e-8:
            print('First order condition met')
            return xk                                                             # Terminate the program

        gprev = g


        ## Step calculation ##
        # Try a Quasi-Newton approximation H = H + (y-H*s)*(y-H*s)/(y-H*s)'*y
        # SR1? BFGS? L-BFGS? L-SR1?

        #sk = CG_Steinhaug_matFree(epsTR, g.reshape((len(g),1)), delta, S, Y)
        if k == 286:
            k = k

        sk, crit = CG_Steinhaug_BFGS_matFree(epsTR, g.reshape((len(g),1)), delta, S, Y)
        #sHs = sk.T@Hessian_times_vec(Y, S, sk)
        sHs = sk.T@BFGS_hessian_times_vec(Y, S, sk)
        model_change = 0.5*sHs + g@sk #m(sk) - m(0)

        sk = sk.reshape(n)

        [fplus,gplus] = fun(xk+sk)

        ## Acceptance of the trial point

        #####NEED TO CHANGE THIS
        function_change = fplus - f
        rhok = function_change / model_change
        if model_change > 0:
            print('Something wrong with TR solver, CG stopped with', crit)
            k=k



        if rhok < n1:
            xnew = xk
        else:
            xnew = xk + sk
            f = fplus
            g = gplus

        if np.isnan(np.sum(xnew)):
            print('Iterate is NaN')
            break

        ## Radius update

        if rhok >= n2:
            delta = min(gamma2*delta, max_delta)
        elif rhok < n1:
            delta = gamma1*delta


        ## Updating Hessian using SR-1 update

        y = gplus-gprev

        if hessian == 1:
            #H, Y, S = update_hessian(H, B0, Y, S, gplus, gprev, sk, memory, sr1_tol)
            #Y, S = updateYS(Y, S, y, sk, memory, sr1_tol)
            Y, S = BFGS_updateYS(Y, S, y, sk, memory, sr1_tol)
        elif hessian == 0:
            H = np.zeros(n,n)

        #fvec = np.hstack(fvec,f)
        fvec.append(f)
        gvec.append(norm(g))
        #gvec = cat(1,gvec,norm(g))
        xk = xnew
        #xvec = cat(2,xvec,xk)
        xvec.append(xk)
        print("k=", k, ", f=", f, ", delta=", delta, "\t ||g||=", norm(g), crit )#= %.8g \n",(k,f,norm(g)))
        if delta <= 1e-16:                                                       # there"s no way you"re making progress at this point
            print('delta is too small')
            return xk


    #xfailed = 13*np.ones(x0.shape)
    #return xfailed



