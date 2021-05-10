import sys
import numpy as np
from numpy.linalg import norm

#sys.path.append('./functions')
from util_func import L_BFGS_two_loop_recursion, CG_Steinhaug_matFree, update_hessian

#function [xvec,fvec,gvec] = TR_fixed(x0,fun,hessian,delta0,eps,budget)
def TR_fixed(x0, fun, hessian, delta0, eps, budget):
    """

    :param x0: initial guess
    :param fun: function handle
    :param hessian: update hessian or not? (I think)
    :param delta0:
    :param eps:
    :param budget: budget constraint (maybe max number of iterations
    :return:
    """

    ## Check for hessian value (seeing if it's 0 or 1, is this just boolean)
    # Not sure what this does. Need Hessian to equal 1 or 0 or else we break
    if not (hessian == 0 or hessian == 1):
        print("hessian value must be either 0 or 1.")
        return


    #X=x0[:]
    n = x0.shape[0]                                                             # dimension of the problem


    ##### INITIALIZATION #####

    n1 = 0.1                                                                  # n1 denotes eta_1 such that 0 < eta_1 <= eta_2 .< 1
    n2 = 0.75                                                                 # n2 denotes eta_2
    gamma1 = 0.5                                                              # 0 < gamma_1 <= gamma_2 .< 1
    gamma2 = 2
    n0 = 0.04*n1                                                              # n0 denotes eta_0 such that eta_0 .< 0.5*(eta_1)
    kg= (0.5*(1-n2)-n0)-1                                                     # kg denotes kappa_g such that kappa_g .< 0.5*(1-eta_2) -eta_0
    epsTR = 1.0e-6

    ## Computing f0 ##

    #xk = x0.reshape((n,1))
    xk = x0

    delta = delta0

    fvec = []    # might want to change since empty lists, not vectors
    gvec = []
    xvec = []
    S = np.zeros(0)
    Y = np.zeros(0)
    B0 = np.eye(n)
    H = np.zeros((n,n))
    memory = 30
    sr1_tol = 1e-4

    #S = np.zeros((n,1))
    #Y = np.zeros((n,1))
    #H = np.eye(n)  #Diagonal(ones(n,n)) # sets hessian to be indentity

    for k in range(budget):

        if k==0:
            ftemp,gtemp = fun(x0) # evaluate function and gradient
            xk = np.zeros(x0.shape)
            f, g = fun(xk)
            #xtemp = 1e-3*np.random.normal(0,1,x0.shape)
            #, gtemp = fun(xtemp)

            H, Y, S = update_hessian(H, B0, Y, S, g, gtemp, -x0, memory, sr1_tol)
            S = S.reshape((S.shape[0],1))
            Y = Y.reshape((Y.shape[0],1))
            #xk = xk.reshape((n,1))

        # check if gradient is sufficiently small
        if np.linalg.norm(g) <= eps/(1+kg):
            return xk                                                             # Terminate the program

        gprev = g


        ## Step calculation ##
        # Try a Quasi-Newton approximation H = H + (y-H*s)*(y-H*s)/(y-H*s)'*y
        # SR1? BFGS? L-BFGS? L-SR1?

        #     [sk,val,~,~,~] = trust(g,zeros(n),delta);
        #try:
        sk = CG_Steinhaug_matFree(epsTR, g.reshape((len(g),1)), delta, S, Y, n)
        val = 0.5*(sk.T@H)@sk + g@sk
        sk = sk.reshape(n)
            #[sk,val,~,~,~] = trust(g,H,delta)
        #except:
        #    break


        #[sk,val,~,~,~] = trust(g,H,delta)

        # This is where precision_level should change.
        # If the single/double/half choice changes here; then you need to
        # recompute f;g at the new precision.

        ## Evaluate the objective function ##

        [fplus,gplus] = fun(xk+sk)

        ## Acceptance of the trial point

        #####NEED TO CHANGE THIS

        rhok = (f-fplus) / (-val)

        if rhok < n1:
            xnew = xk
        else:
            xnew = xk + sk
            f = fplus
            g = gplus


        ## Radius update

        if rhok >= n2:
            delta = gamma2*delta
        elif rhok <= n1:
            delta = gamma1*delta


        ## Updating Hessian using SR-1 update

        y = gplus-gprev

        if hessian == 1:
            H, Y, S = update_hessian(H, B0, Y, S, gplus, gprev, sk, memory, sr1_tol)
        elif hessian == 0:
            H = np.zeros(n,n)
        """
        if hessian == 1
            grad_pred = y-H.*sk
            dp = (grad_pred)'.*sk
            update = ((grad_pred).*(grad_pred)'./dp)
            if abs(dp) >= 1e-8*norm(grad_pred)*norm(sk) and (not any(any(isnan(update))) and not any(any(isinf(update)))):
                H = H + update
        """




        #fvec = np.hstack(fvec,f)
        fvec.append(f)
        gvec.append(norm(g))
        #gvec = cat(1,gvec,norm(g))
        xk = xnew
        #xvec = cat(2,xvec,xk)
        xvec.append(xk)
        print("k = %g, f = %g, norm(g) = %.8g \n",(k,f,norm(g)))
        if delta <= 1e-16:                                                       # there"s no way you"re making progress at this point
            break




