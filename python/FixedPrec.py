import sys
import numpy as np
from numpy.linalg import norm
import util_func_v2

#sys.path.append('./functions')
from util_func_v2 import CG_Steinhaug_matFree, Hessian_times_vec, updateYS, BFGS_CG_Steinhaug_matFree, BFGS_hessian_times_vec, BFGS_updateYS

def TR_fixed(x0, fun, eps, max_iter, verbose=False, max_memory=30, solver='sr1', epsTR=1.e-7, store_history=False, delta_init=None):
    """
    :param x0: initial guess
    :param fun: function handle
    :param delta0: initial trust region radius
    :param eps: problem tolerance
    :param max_iter: max_iter constraint (maybe max number of iterations
    :param epsTR: residual tolerance for trust region subproblem
    :return: search direction for trust region subproblem
    """

    ##### INITIALIZATION #####
    xk = x0
    n = x0.shape[0]                                                             # dimension of the problem
    max_delta = 1e4
    memory = min(max_memory, len(x0))
    sr1_tol = 1e-4
    n1 = 0.01                                                                  # n1 denotes eta_1 such that 0 < eta_1 <= eta_2 .< 1
    n2 = 0.1                                                                # n2 denotes eta_2
    gamma1 = 0.5                                                              # 0 < gamma_1 <= gamma_2 .< 1
    gamma2 = 2
    gamma = 1
    fvec = []    # might want to change since empty lists, not vectors
    gvec = []
    xvec = []
    S = []
    Y = []
    func_count = 0
    machine_eps = np.finfo(float).eps

    for k in range(max_iter):
        if k == 0:
            f,g = fun(xk)
            delta = norm(g) if delta_init is None else delta_init
            func_count += 1

        # check if gradient is sufficiently small or TR radius is below machine precision
        if np.linalg.norm(g) <= eps:
            message='First order condition met'
            success=True
            break
            if verbose: print(message)
        if delta <= np.sqrt(machine_eps):
            message='TR radius too small'
            success=False
            if verbose: print(message)
            break                                                           # Terminate the program

        gprev = g


        ## Step calculation ##
        # Try a Quasi-Newton approximation H = H + (y-H*s)*(y-H*s)/(y-H*s)'*y
        # SR1? BFGS? L-BFGS? L-SR1?

        # SR1 update
        if solver=='sr1':
            sk, crit = CG_Steinhaug_matFree(epsTR, g.reshape((n,1)), delta, S, Y, gamma, verbose=verbose)
            sHs = sk.T@Hessian_times_vec(Y, S, gamma, sk)
        elif solver=='lbfgs':
            sk, crit = BFGS_CG_Steinhaug_matFree(epsTR, g.reshape((n,1)), delta, S, Y, gamma)
            sHs = sk.T@BFGS_hessian_times_vec(Y, S, gamma, sk)
        else:
            print('Solver not recognized')
            break


        model_change = np.sum(0.5*sHs + g@sk)  #m(sk) - m(0).... sum just turns this into a scalar
        sk = sk.reshape(n)
        [fplus,gplus] = fun(xk+sk)
        func_count += 1

        ## Acceptance of the trial point
        # we expect function to decrease, but may not if model function isn't good
        function_change = fplus - f
        rhok = function_change / model_change
        if model_change > 0 and verbose:
            print("\n \n \nModel increase by", model_change, " and ||pk||=", norm(sk))

        if rhok < n1 or model_change > 0:
            xnew = xk
        else:
            xnew = xk + sk
            f = fplus
            g = gplus

        if np.isnan(np.sum(xnew)):
            if verbose: print('Iterate is NaN')
            message="Returned NaN"
            success=False
            break
        if np.isinf(abs(fplus)):
            if verbose: print('Function is +/- inf')
            message="Returned =/- inf"
            success=False
            break


        ## Radius update

        if rhok >= n2 and abs(rhok) != np.inf and model_change < 0:
            if norm(sk) > .8*delta:
                delta = min(gamma2*delta, max_delta)
        else:
            if (rhok < n1) or model_change >= 0:
                delta = gamma1*delta

        ## Updating Hessian using SR-1 update or BFGS
        y = gplus-gprev
        if solver=='sr1':
            Y, S = updateYS(Y, S, y, sk, memory, gamma, sr1_tol=sr1_tol, verbose=verbose)
        elif solver=='lbfgs':
            Y,S = BFGS_updateYS(Y, S, y, sk, memory, gamma, sr1_tol=sr1_tol)
        else:
            print('Solver not recognized')


        xk = xnew
        if store_history:
            xvec.append(xk)
            fvec.append(f)
            gvec.append(norm(g))

        if verbose: print("k=", k, ", f=", f, ", delta=", delta, "\t ||g||=", norm(g), crit )

        #
        if delta <= 1e-32:
            if verbose: print('Delta is too small')
            message="TR radius too small"
            success=False
            break

    if k == (max_iter-1):
        message="Exceed max iterations"
        success=False

    ret = util_func_v2.structtype(x=xk, fun=f, jac=g, message=message, success=success, nfev=func_count, nit=k)
    return ret







