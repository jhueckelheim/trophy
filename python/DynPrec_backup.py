from scipy import optimize
import numpy as np
from numpy.linalg import norm
import util_func_v2

"""
TO DO: Several places where indices may need to be corrected and must make changes for calls to TRS
"""


def DynTR(x0, fun, prec_vec, epsilon, budget, verbose=False, max_memory=30):
    """
    :param x0: numpy array, initialization
    :param fun:  objective/gradient fun
    :param prec_vec: vector of precision values
    :param epsilon: final gradient accuracy
    :param budget: budget constraint
    :return:
    """
    # Later: Check that prec_vec is listed in ascending order

    num_prec = max(prec_vec)
    prec_lvl = 0                        # MIGHT NEED TO ADJUST INDEXING
    prec = prec_vec[prec_lvl]
    func_counter = 0
    prec_lvl_counter = [0,0]

    # Set the initial iterate
    x = x0
    n = x.shape[0]
    machine_eps = np.finfo(float).eps

    # Internal algorithmic parameters: Should expose through a struct Type
    eta_good = 0.01
    eta_great = 0.1
    gamma_dec = 0.5
    gamma_inc = 2
    sr1_tol = 1.0e-4

    # First function evaluation:
    # print("Function evaluations at precision level ", prec, " bits:")
    f,g = fun(x,prec)
    prec_lvl_counter[prec_lvl] += 1

    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    delta = norm(g)
    max_delta = max(1.0e4, norm(g))

    # Initialize Hessian - max_memory should be exposed
    memory = min(max_memory,n+1)
    S = []
    Y = []
    B0 = np.eye(n)
    H = np.zeros((n,n))

    # benchmarking:
    xhist = list() #zeros((budget+1,n), dtype='np.float64')
    prechist = str()

    first_fail = True
    first_success = 0
    theta = 0.0
    epsTR = 1.0e-7
    gamma = 1

    for i in range(budget):   # note that this indexes from 0 and not 1, check for errors because of it
        # benchmarking:
        prec_str = ""

        # criticality check
        if (norm(g) <= epsilon) or (delta <= np.sqrt(machine_eps)):
            if prec == num_prec:
                if norm(g) <= epsilon:
                    # terminate, because we're done
                    message='First order condition met'
                    success=True
                    if verbose: print(message)
                else:
                    message='TR radius too small'
                    success=False
                    if verbose: print(message)
                break
            else:
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                if verbose: print("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f,g = fun(x,prec)
                prec_lvl_counter[prec_lvl] += 1
                # hard-coded benchmarking:
                if prec_lvl == 1:
                    prec_str = prec_str + "s"
                elif prec_lvl == 2:
                    prec_str = prec_str + "d"


        s, crit = util_func_v2.CG_Steinhaug_matFree(epsTR, g, delta, S, Y, gamma, verbose=False)
        predicted_reduction = sum(-0.5*(s.T@util_func_v2.Hessian_times_vec(Y,S,gamma,s)) - s.T@g)  # this should be greater than zero
        s = s.reshape((n,))
        if predicted_reduction <= 0:
            if predicted_reduction == 0:
                # I think this is the result of having the same exact subproblem after update since we have
                # two versions of same function for different precisions.
                if verbose: print('No predicted reduction')
                predicted_reduction = np.inf
            else:
                if verbose: print('Step gives model function increase of', -predicted_reduction)


        # evaluate function at current precision at new trial point
        fplus, gplus = fun(x+s,prec)
        prec_lvl_counter[prec_lvl] += 1

        # hard-coded benchmarking:
        if prec == 1:
            prec_str = prec_str + "s"
        elif prec == 2:
            prec_str = prec_str + "d"

        # determine acceptance of trial point
        actual_reduction = f - fplus   # should be greater than zero
        rho = actual_reduction/predicted_reduction

        # for Hessian updating:
        gprev = g

        if (rho < eta_good) or (predicted_reduction < 0):
            # the iteration was a failure
            xnew = x

            # do we update the precision?
            """
            if first_fail:
                first_fail = False
                if prec_lvl < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[prec_lvl + 1]
                    print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:")
                    ftemp,gtemp = fun(x,temp_prec)
                    ftempplus,gtempplus = fun(x+s,temp_prec)
                    
                    # hard-coded benchmarking, added a second "s" and "d" since we have two function evals at higher lvl
                    if prec_lvl + 1 == 1:
                        prec_str = string(prec_str, "ss")
                    elif prec_lvl + 1 == 2:
                        prec_str = prec_str + "dd"
                    

                    # initial theta (how different are evals at different precision, if big, increase precision)
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if isnan(theta):
                        theta = sqrt(machine_eps)

                    if theta > eta_good*predicted_reduction and delta < min(1.0,norm(gtemp)):
                    #if (theta > eta_good*(-g@s - 0.5*(s@H)@s)) and (delta < min(1.0,norm(gtemp))):
                        # update precision
                        prec_lvl += 1
                        print("Permanently switching to precision level ", temp_prec, " bits:")
                        f = ftemp
                        g = gtemp
                        gplus = gtemp
                        delta = min(norm(g),1.0)

                        xnew = x + s
                        gprev = gtemp

                    else:
                        # just shrink the TR
                        delta = gamma_dec*norm(s)

                else: # we can't increase the precision anymore, so just reject
                    delta = gamma_dec*norm(s)  # should this be current trust region radius rather that s? I guess s is smaller
            """

            if first_fail:
                first_fail = False
                if prec < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[-1] #take highest precision (originally set to next highest but this agrees with algo
                    if verbose: print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:")
                    ftemp,gtemp = fun(x,temp_prec)
                    ftempplus,gtempplus = fun(x+s,temp_prec)
                    prec_lvl_counter[-1] += 2

                    # initial theta (how different are evals at different precision, if big, increase precision)
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if np.isnan(theta):
                        theta = np.sqrt(machine_eps)


            # reason for else is that if we don't have this, we will need to evaluate again, not sure that's a big concern
            if theta > eta_good*predicted_reduction and prec < num_prec and delta < min(1.0,norm(g)):
                temp_prec = prec_vec[prec_lvl + 1]
                if verbose: print("Probed a pair of function evaluations at ", temp_prec, " bits...")
                ftemp, gtemp = fun(x,temp_prec)
                ftempplus,gtempplus = fun(x+s,temp_prec)
                prec_lvl_counter[prec_lvl+1] += 2

                # hard-coded benchmarking:
                if prec_lvl + 1 == 1:
                    prec_str = prec_str + "ss"
                elif prec_lvl + 1 == 2:
                    prec_str = prec_str + "dd"


                theta = abs((f-fplus)-(ftemp-ftempplus))
                if theta > eta_good*predicted_reduction:
                    prec_lvl += 1
                    prec = prec_vec[prec_lvl]
                    if verbose: print("Permanently switching to precision level ", temp_prec, " bits:")
                    f = ftemp
                    g = gtemp
                    fplus = ftempplus
                    gplus = gtempplus
                    gprev = gtemp
                    delta = min(1.0,norm(g))
            else: # the model quality isn't suspect, standard reject
                delta = gamma_dec*delta

        else: # the step is at least good
            if first_success == 0:
                first_success = 1

            xnew = x + s
            f = fplus
            g = gplus

            # is the step great (do we get desired reduction from model and are we close to TR radius)?
            if (norm(s) > 0.8*delta) and (rho > eta_great):
                delta = min(max_delta,gamma_inc*delta)
            else:
                #delta = norm(s)   # why would we shrink trust region if successful????
                delta = delta     #replacing above line with this since it's what algo states.

        y = gplus-gprev
        if first_success == 1:
            #constant = (y0'*s)/(y0'*y0)    # originally in code. I believe this is for inverse hessian
            #constant = 1.0

            # following choice is based on B_0 for initial approximate Hessian as discussed on pp. 178 and 182 N&W
            #gamma = norm(y)**2/(y.T@s)    # reciprocals of one another, which is more appropriate for approx Hessian?


            B0 = gamma*B0
            first_success = 2

        # update the Hessian
        if first_success > 0:
            Y, S = util_func_v2.updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)


        # get ready for next iteration
        x = xnew

        # TODO: write better output to command line indicating progress
        if verbose: print("iter: ", i+1, ", f: ",f, ", delta: ", delta, ", norm(g): ", norm(g), ' stopping criteria:', crit)
        if np.isnan(f) or np.isinf(f):
            message='Obj is nan or +/-inf'
            success=False
            if verbose: print('Obj is nan or +/-inf')
            break

        # early termination if progress seems unlikely
        if norm(g) < epsilon:
            if prec == num_prec:
                message='First order condition met'
                success=True
                if verbose: print(message)
                break


    if i == (budget-1):
        message="Exceed max iterations"
        success=False

    ret = util_func_v2.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(prec_lvl_counter), nit=i)

    return ret

