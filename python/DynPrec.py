from scipy import optimize
import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm
from numpy.linalg import norm
import util_func_v2
import time

def DynTR(x0, fun, prec_vec, gtol=1.0e-5, max_iter=1000, verbose=False, max_memory=30, store_history=False,
          hessian_updates='sr1', tr_tol=1.e-5, delta_init=None, max_delta=1e4, sr1_tol=1.e-4):
    """
    :param x0: numpy array, initialization
    :param fun:  objective/gradient function
    :param prec_vec: vector of precision values, i.e., 1/2, 1, 2. Might be bugs that come up with more than 2 precisions
    :param gtol: final gradient norm accuracy tolerance
    :param max_iter: maximum number of iterations allowed 
    :param verbose: return out put or not (default False)
    :param max_memory: number of curvature pairs to store (default 30)
    :param store_history: return entire history of evaluations
    :param hessian_updates: use bfgs updates or sr1 (default)
    :param tr_tol: tolerance for trust region subproblem (default 1e-07)

    :return:
    """
    # Later: Check that prec_vec is listed in ascending order

    # set highest level of precision and counters to zero
    num_prec = max(prec_vec)
    prec_lvl = 0
    prec = prec_vec[prec_lvl]
    prec_lvl_counter = [0, 0]

    # Set the initial iterate
    x = x0
    n = x.shape[0]
    machine_eps = np.finfo(float).eps

    # Internal algorithmic parameters: Should expose through a struct Type
    eta_good = 0.01
    eta_great = 0.1
    gamma_dec = 0.5
    gamma_inc = 2

    # First function evaluation:
    f, g = fun(x, prec)
    prec_lvl_counter[prec_lvl] += 1

    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    delta = norm(g) if delta_init is None else delta_init
    max_delta = max(max_delta, norm(g))

    # Initialize Hessian - max_memory should be exposed
    memory = min(max_memory, n+1)
    S = []
    Y = []

    first_fail = True
    first_success = 0
    theta = 0.0
    gamma = 1

    for i in range(max_iter):
        # criticality check
        if (norm(g) <= gtol) or (delta <= np.sqrt(machine_eps)):
            if prec == num_prec:
                if norm(g) <= gtol:
                    # terminate, because we're done
                    message='First order condition met'
                    success = True
                    print(message) if verbose else None
                else:
                    message = 'TR radius too small'
                    success = False
                    print(message) if verbose else None
                break
            else:
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                print("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f, g = fun(x, prec)
                prec_lvl_counter[prec_lvl] += 1
                if norm(g) <= gtol:
                    # terminate, because we're done
                    message = 'First order condition met'
                    success = True
                    print(message) if verbose else None


        if hessian_updates == 'sr1':
            s, crit = util_func_v2.CG_Steinhaug_matFree(tr_tol, g, delta, S, Y, gamma, verbose=False)
            predicted_reduction = np.sum(-0.5*(np.matmul(s.T,util_func_v2.Hessian_times_vec(Y,S,gamma,s))) - np.dot(s.T,g))  # this should be greater than zero
        elif hessian_updates == 'lbfgs':
            s, crit = util_func_v2.BFGS_CG_Steinhaug_matFree(tr_tol, g.reshape((n,1)), delta, S, Y, gamma)
            predicted_reduction = np.sum(-0.5*np.matmul(s.T, util_func_v2.BFGS_hessian_times_vec(Y,S,gamma,s)) - np.dot(s.T@g))  # this should be greater than zerosum(-0.5)
        else:
            print('Update not recognized')
            break

        s = s.reshape((n,))
        if predicted_reduction <= 0:
            if predicted_reduction == 0:
                # I think this is the result of having the same exact subproblem after update since we have
                # two versions of same function for different precisions.
                print('No predicted reduction') if verbose else None
                predicted_reduction = np.inf
            else:
                print('Step gives model function increase of', -predicted_reduction) if verbose else None

        # evaluate function at current precision at new trial point
        fplus, gplus = fun(x+s, prec)
        prec_lvl_counter[prec_lvl] += 1


        # determine acceptance of trial point
        actual_reduction = f - fplus   # should be greater than zero
        rho = actual_reduction/predicted_reduction

        # for Hessian updating:
        gprev = g

        if (rho < eta_good) or (predicted_reduction < 0):
            # the iteration was a failure
            xnew = x

            # do we update the precision?
            if first_fail:
                first_fail = False
                if prec < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[-1] #take highest precision (originally set to next highest but this agrees with algo
                    print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:") if verbose else None
                    ftemp, gtemp = fun(x,temp_prec)
                    ftempplus, gtempplus = fun(x+s, temp_prec)
                    #print('Evaluating float64', ftemp.dtype)
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

                theta = abs((f-fplus)-(ftemp-ftempplus))
                if theta > eta_good*predicted_reduction:
                    prec_lvl += 1
                    prec = prec_vec[prec_lvl]
                    print("Permanently switching to precision level ", temp_prec, " bits:")
                    f = ftemp
                    g = gtemp
                    fplus = ftempplus
                    gplus = gtempplus
                    gprev = gtemp
                    delta = min(1.0, norm(g))
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
                delta = min(max_delta, gamma_inc*delta)
            else:
                delta = delta     #replacing above line with this since it's what algo states.

        y = gplus-gprev
        if first_success == 1:
            #constant = (y0'*s)/(y0'*y0)    # originally in code. I believe this is for inverse hessian
            #constant = 1.0

            # following choice is based on B_0 for initial approximate Hessian as discussed on pp. 178 and 182 N&W
            #gamma = norm(y)**2/(y.T@s)    # reciprocals of one another, which is more appropriate for approx Hessian?

            first_success = 2

        # update the Hessian
        if first_success > 0:
            if hessian_updates == 'sr1':
                Y, S = util_func_v2.updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)
            elif hessian_updates == 'lbfgs':
                Y, S = util_func_v2.BFGS_updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)

        # get ready for next iteration
        x = xnew
        x_hist.append(x) if store_history else None

        # TODO: write better output to command line indicating progress
        if verbose:
            print("iter: ", i+1, ", f: ", f, ", delta: ", delta, ", norm(g): ", norm(g),
                  ' stopping criteria:', crit, "||g||_inf:", np.max(np.abs(g)))
        if np.isnan(f) or np.isinf(f):
            message = 'Obj is nan or +/-inf'
            success = False
            print('Obj is nan or +/-inf') if verbose else None
            break

        # early termination if progress seems unlikely
        if norm(g) < gtol:
            if prec == num_prec:
                message = 'First order condition met'
                success = True
                print(message) if verbose else None
                break


    if i == (max_iter-1):
        message = "Exceed max iterations"
        success = False

    ret = util_func_v2.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(prec_lvl_counter),
                                  nit=i, precision_counts=prec_lvl_counter)

    return ret






def DynTR_for_pydda(x0, fun, prec_vec, gtol=1.0e-5, max_iter=1000, verbose=False, max_memory=30, store_history=False,
          hessian_updates='sr1', tr_tol=1e-6, delta_init=None, max_delta=1e4, sr1_tol=1.e-4):
    """
    :param x0: numpy array, initialization
    :param fun:  objective/gradient function
    :param prec_vec: vector of precision values, i.e., 1/2, 1, 2. Might be bugs that come up with more than 2 precisions
    :param gtol: final gradient norm accuracy tolerance
    :param max_iter: maximum number of iterations allowed
    :param verbose: return out put or not (default False)
    :param max_memory: number of curvature pairs to store (default 30)
    :param store_history: return entire history of evaluations
    :param hessian_updates: use bfgs updates or sr1 (default)
    :param tr_tol: tolerance for trust region subproblem (default 1e-07)

    :return:
    """
    # Later: Check that prec_vec is listed in ascending order

    # set highest level of precision and counters to zero
    #if tr_tol is None:
    #    tr_tol = gtol/10
    nprec = len(prec_vec)
    num_prec = max(prec_vec)
    prec_lvl = 0
    prec = prec_vec[prec_lvl]
    prec_lvl_counter = [0, 0]
    f_hist = list()
    prec_hist = list()
    inf_norm_hist = list()
    two_norm_hist = list()
    single_array = list()
    double_array = list()
    textra = 0

    # Set the initial iterate
    x = x0
    n = x.shape[0]
    machine_eps = np.finfo(float).eps

    # Internal algorithmic parameters: Should expose through a struct Type
    eta_good = 0.01
    eta_great = 0.1
    gamma_dec = 0.5
    gamma_inc = 2

    # First function evaluation:
    f, g = fun(x, prec)
    prec_lvl_counter[prec_lvl] += 1
    if store_history:
        f_hist.append(f)
        prec_hist.append(prec_lvl)
        inf_norm_hist.append(norm(g, np.inf))
        two_norm_hist.append(norm(g))




    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    delta = norm(g) if delta_init is None else delta_init
    max_delta = max(max_delta, norm(g))

    # Initialize Hessian - max_memory should be exposed
    memory = min(max_memory, n+1)
    S = []
    Y = []

    first_fail = True
    first_success = 0
    theta = 0.0
    gamma = 1

    for i in range(max_iter):
        # criticality check
        if norm(g) <= gtol or delta <= np.sqrt(machine_eps) or norm(g,np.inf) <= gtol:
            if prec == num_prec:
                if norm(g) <= gtol or norm(g, np.inf) <= gtol:
                    # terminate, because we're done
                    message='First order condition met'
                    success = True
                    print(message) if verbose else None
                else:
                    message = 'TR radius too small'
                    success = False
                    print(message) if verbose else None
                break
            else:
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                print("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f, g = fun(x, prec)
                if store_history:
                    f_hist.append(f)
                    prec_hist.append(prec_lvl)
                    inf_norm_hist.append(norm(g, np.inf))
                    two_norm_hist.append(norm(g))


                prec_lvl_counter[prec_lvl] += 1
                if norm(g) <= gtol or norm(g, np.inf) <= gtol:
                    # terminate, because we're done
                    message = 'First order condition met'
                    success = True
                    print(message) if verbose else None


        if hessian_updates == 'sr1':
            s, crit = util_func_v2.CG_Steinhaug_matFree(tr_tol, g, delta, S, Y, gamma, verbose=False, max_it=10*max_memory)
            predicted_reduction = np.sum(-0.5*(np.matmul(s.T,util_func_v2.Hessian_times_vec(Y,S,gamma,s))) - np.dot(s.T,g))  # this should be greater than zero
        elif hessian_updates == 'lbfgs':
            s, crit = util_func_v2.BFGS_CG_Steinhaug_matFree(tr_tol, g.reshape((n,1)), delta, S, Y, gamma)
            predicted_reduction = np.sum(-0.5*np.matmul(s.T, util_func_v2.BFGS_hessian_times_vec(Y,S,gamma,s)) - np.dot(s.T@g))  # this should be greater than zerosum(-0.5)
        else:
            print('Update not recognized')
            break

        s = s.reshape((n,))
        if predicted_reduction <= 0:
            if predicted_reduction == 0:
                # I think this is the result of having the same exact subproblem after update since we have
                # two versions of same function for different precisions.
                print('No predicted reduction') if verbose else None
                predicted_reduction = np.inf
            else:
                print('Step gives model function increase of', -predicted_reduction) if verbose else None

        # evaluate function at current precision at new trial point
        fplus, gplus = fun(x+s, prec)
        prec_lvl_counter[prec_lvl] += 1


        # determine acceptance of trial point
        actual_reduction = f - fplus   # should be greater than zero
        rho = actual_reduction/predicted_reduction

        # for Hessian updating:
        gprev = g

        if (rho < eta_good) or (predicted_reduction < 0):
            # the iteration was a failure
            xnew = x

            # do we update the precision?
            if first_fail:
                first_fail = False
                if prec < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[-1] #take highest precision (originally set to next highest but this agrees with algo
                    print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:") if verbose else None
                    ftemp, gtemp = fun(x,temp_prec)
                    ftempplus, gtempplus = fun(x+s, temp_prec)
                    #print('Evaluating float64', ftemp.dtype)
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
                ftempplus, gtempplus = fun(x+s, temp_prec)
                prec_lvl_counter[prec_lvl+1] += 2

                theta = abs((f-fplus)-(ftemp-ftempplus))
                if theta > eta_good*predicted_reduction:
                    prec_lvl += 1
                    prec = prec_vec[prec_lvl]
                    print("Permanently switching to precision level ", temp_prec, " bits:")
                    f = ftemp
                    g = gtemp
                    gplus = gtempplus
                    gprev = gtemp
                    delta = min(1.0, norm(g))
            else: # the model quality isn't suspect, standard reject
                delta = gamma_dec*delta

        else: # the step is at least good
            if first_success == 0:
                first_success = 1

            xnew = x + s
            f = fplus
            g = gplus

            # will only take function value for successful steps
            if store_history:
                f_hist.append(np.ndarray.item(np.array(f)))
                prec_hist.append(prec_lvl)
                norm_g_inf = norm(g, np.inf)
                norm_g_two = norm(g)
                inf_norm_hist.append(norm_g_inf)
                two_norm_hist.append(norm_g_two)

                if nprec > 1:
                    t1 = time.time()
                    # is the current evaluation a single?
                    if prec_lvl == 0:
                        # if so, calculate double function and grad and append
                        fd, gd = fun(xnew, 2)
                        single_array.append([np.ndarray.item(np.array(f)), norm_g_two, norm_g_inf])
                        double_array.append([fd, norm(gd), norm(gd, np.inf)])
                    if prec_lvl == 1:
                        fs, gs = fun(xnew, 1)
                        single_array.append([fs, norm(gs), norm(gs, np.inf)])
                        double_array.append([np.ndarray.item(np.array(f)), norm_g_two, norm_g_inf])
                    t2 = time.time()
                    textra += (t2-t1)


            # is the step great (do we get desired reduction from model and are we close to TR radius)?
            if (norm(s) > 0.8*delta) and (rho > eta_great):
                delta = min(max_delta, gamma_inc*delta)
            else:
                delta = delta     #replacing above line with this since it's what algo states.

        y = gplus-gprev
        if first_success == 1:
            #constant = (y0'*s)/(y0'*y0)    # originally in code. I believe this is for inverse hessian
            #constant = 1.0

            # following choice is based on B_0 for initial approximate Hessian as discussed on pp. 178 and 182 N&W
            #gamma = norm(y)**2/(y.T@s)    # reciprocals of one another, which is more appropriate for approx Hessian?

            first_success = 2

        # update the Hessian
        if first_success > 0:
            Y, S = util_func_v2.updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)


        # get ready for next iteration
        x = xnew
        #x_hist.append(x) if store_history else None

        # TODO: write better output to command line indicating progress
        if verbose:
            print("iter: ", i+1, ", f: ", f, ", delta: ", delta, ", norm(g): ", norm(g),
                  ' stopping criteria:', crit, "||g||_inf:", norm(g, np.inf))
        if np.isnan(f) or np.isinf(f):
            message = 'Obj is nan or +/-inf'
            success = False
            print('Obj is nan or +/-inf') if verbose else None
            break

        # early termination if progress seems unlikely
        if norm(g) < gtol or norm(g, np.inf) < gtol:
            if prec == num_prec:
                message = 'First order condition met'
                success = True
                print(message) if verbose else None
                break


    if i == (max_iter-1):
        message = "Exceed max iterations"
        success = False

    ret = util_func_v2.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(prec_lvl_counter),
                                  nit=i, precision_counts=prec_lvl_counter)
    if store_history:
        ret.f_hist = np.array(f_hist)
        ret.prec_hist = np.array(prec_hist)
        ret.g_inf_norm_hist = np.array(inf_norm_hist)
        ret.g_two_norm_hist = np.array(two_norm_hist)
        if nprec > 1:
            ret.single_array = np.array(single_array)
            ret.double_array = np.array(double_array)
            ret.textra = textra

    return ret







'''
def DynTR_JAX(x0, fun, prec_vec, gtol=1e-5, max_iter=1000, verbose=False, max_memory=30, store_history=False,
          hessian_updates='sr1', tr_tol=1.e-5, delta_init=None, max_delta=1e4, sr1_tol=1.e-4):
    """
    :param x0: numpy array, initialization
    :param fun:  objective/gradient function
    :param prec_vec: vector of precision values, i.e., 1/2, 1, 2. Might be bugs that come up with more than 2 precisions
    :param gtol: final gradient norm accuracy tolerance
    :param max_iter: maximum number of iterations allowed
    :param verbose: return out put or not (default False)
    :param max_memory: number of curvature pairs to store (default 30)
    :param store_history: return entire history of evaluations
    :param hessian_updates: use bfgs hessian_updates or sr1 (default)
    :param tr_tol: tolerance for trust region subproblem (default 1e-07)

    :return:
    """
    # Later: Check that prec_vec is listed in ascending order

    # set highest level of precision and counters to zero
    num_prec = max(prec_vec)
    prec_lvl = 0
    prec = prec_vec[prec_lvl]
    prec_lvl_counter = [0, 0]

    # Set the initial iterate
    x = jnp.array(x0)
    n = x.shape[0]
    machine_eps = np.finfo(float).eps

    # Internal algorithmic parameters: Should expose through a struct Type
    eta_good = 0.01
    eta_great = 0.1
    gamma_dec = 0.5
    gamma_inc = 2

    # First function evaluation:
    f, g = fun(x, prec)
    g = jnp.array(g)
    g_norm = jnorm(g)
    prec_lvl_counter[prec_lvl] += 1

    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    delta = g_norm if delta_init is None else delta_init
    max_delta = max(max_delta, g_norm)

    # Initialize Hessian - max_memory should be exposed
    memory = min(max_memory, n+1)
    S = []
    Y = []

    first_fail = True
    first_success = 0
    theta = 0.0
    gamma = 1

    for i in range(max_iter):
        # criticality check
        if (g_norm <= gtol) or (delta <= np.sqrt(machine_eps)):
            if prec == num_prec:
                if g_norm <= gtol:
                    # terminate, because we're done
                    message='First order condition met'
                    success=True
                    if verbose: print(message)
                else:
                    message = 'TR radius too small'
                    success = False
                    if verbose: print(message)
                break
            else:
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                print("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f, g = fun(x, prec)
                prec_lvl_counter[prec_lvl] += 1
                if jnorm(g) <= gtol:
                    # terminate, because we're done
                    message = 'First order condition met'
                    success = True
                    if verbose: print(message)


        if hessian_updates == 'sr1':
            s, crit = util_func_v2.CG_Steinhaug_matFree_JAX(tr_tol, g, delta, S, Y, gamma, verbose=False)
            predicted_reduction = jnp.sum(-0.5*(jnp.dot(s.T, util_func_v2.Hessian_times_vec_JAX(Y, S, gamma, s))) - jnp.dot(s.T, g))  # this should be greater than zero
        else:
            print('Solver not recognized')
            break

        s = s.reshape((n,))
        if predicted_reduction <= 0:
            if predicted_reduction == 0:
                # I think this is the result of having the same exact subproblem after update since we have
                # two versions of same function for different precisions.
                if verbose: print('No predicted reduction')
                predicted_reduction = jnp.inf
            else:
                if verbose: print('Step gives model function increase of', -predicted_reduction)

        # evaluate function at current precision at new trial point
        fplus, gplus = fun(x+s, prec)
        print(fplus.dtype)
        prec_lvl_counter[prec_lvl] += 1


        # determine acceptance of trial point
        actual_reduction = f - fplus   # should be greater than zero
        rho = actual_reduction/predicted_reduction

        # for Hessian updating:
        gprev = g
        g_norm = jnorm(g)

        if (rho < eta_good) or (predicted_reduction < 0):
            # the iteration was a failure
            xnew = x

            # do we update the precision?
            if first_fail:
                first_fail = False
                if prec < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[-1] #take highest precision (originally set to next highest but this agrees with algo
                    if verbose: print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:")
                    ftemp, gtemp = fun(x,temp_prec)
                    ftempplus, gtempplus = fun(x+s,temp_prec)
                    print('Evaluating float64', ftemp.dtype)
                    prec_lvl_counter[-1] += 2

                    # initial theta (how different are evals at different precision, if big, increase precision)
                    theta = jnp.abs((f-fplus)-(ftemp-ftempplus))
                    if jnp.isnan(theta):
                        theta = jnp.sqrt(machine_eps)


            # reason for else is that if we don't have this, we will need to evaluate again, not sure that's a big concern
            if theta > eta_good*predicted_reduction and prec < num_prec and delta < jnp.min(1.0,jnorm(g)):
                temp_prec = prec_vec[prec_lvl + 1]
                if verbose: print("Probed a pair of function evaluations at ", temp_prec, " bits...")
                ftemp, gtemp = fun(x,temp_prec)
                ftempplus,gtempplus = fun(x+s, temp_prec)
                prec_lvl_counter[prec_lvl+1] += 2

                theta = jnp.abs((f-fplus)-(ftemp-ftempplus))
                if theta > eta_good*predicted_reduction:
                    prec_lvl += 1
                    prec = prec_vec[prec_lvl]
                    print("Permanently switching to precision level ", temp_prec, " bits:")
                    #if verbose: print("Permanently switching to precision level ", temp_prec, " bits:")
                    f = ftemp
                    g = gtemp
                    fplus = ftempplus
                    gplus = gtempplus
                    gprev = gtemp
                    delta = jnp.min(1.0, g_norm)
            else: # the model quality isn't suspect, standard reject
                delta = gamma_dec*delta

        else: # the step is at least good
            if first_success == 0:
                first_success = 1

            xnew = x + s
            f = fplus
            g = gplus

            # is the step great (do we get desired reduction from model and are we close to TR radius)?
            if (jnorm(s) > 0.8*delta) and (rho > eta_great):
                delta = min(max_delta, gamma_inc*delta)
            else:
                delta = delta     #replacing above line with this since it's what algo states.

        y = gplus - gprev
        if first_success == 1:
            #constant = (y0'*s)/(y0'*y0)    # originally in code. I believe this is for inverse hessian
            #constant = 1.0

            # following choice is based on B_0 for initial approximate Hessian as discussed on pp. 178 and 182 N&W
            #gamma = norm(y)**2/(y.T@s)    # reciprocals of one another, which is more appropriate for approx Hessian?
            first_success = 2

        # update the Hessian
        if first_success > 0:
            if hessian_updates == 'sr1':
                Y, S = util_func_v2.updateYS_JAX(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)
            elif hessian_updates == 'lbfgs':
                Y, S = util_func_v2.BFGS_updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol)

        # get ready for next iteration
        x = xnew
        if store_history: x_hist.append(x)

        # TODO: write better output to command line indicating progress
        if verbose: print("iter: ", i+1, ", f: ",f, ", delta: ", delta, ", norm(g): ", g_norm, ' stopping criteria:', crit)
        if jnp.isnan(f) or jnp.isinf(f):
            message='Obj is nan or +/-inf'
            success=False
            if verbose: print('Obj is nan or +/-inf')
            break

        # early termination if progress seems unlikely
        if g_norm < gtol:
            if prec == num_prec:
                message = 'First order condition met'
                success = True
                print(message) if verbose else None
                break


    if i == (max_iter-1):
        message = "Exceed max iterations"
        success = False

    ret = util_func_v2.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(prec_lvl_counter),
                                  nit=i, precision_counts=prec_lvl_counter)

    return ret
'''