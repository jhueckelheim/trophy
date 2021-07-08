import numpy as np
import pandas as pd

#import pycutest  # this probably needs to be changed to the local version rather
import pycutest_for_trophy as pycutest
import FixedPrec
import DynPrec_old_version as DynPrec
import scipy
import sys
import os
import time
from util_func import pycutest_wrapper
from numpy.linalg import norm

# add pycutestcache path if you haven't done so elsewhere
temp = os.environ['PYCUTEST_CACHE']
cache_dir = temp + '/pycutest_cache_holder/'
sys.path.append(temp)
sys.path.append(cache_dir)


#def DynTR(gamma=None,
#          verbose=False, store_history=False, write_folder=None):

ret = util_func.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(precision_counter.values()),
                           nit=i, precision_counts=precision_counter, time_counter=time_counter)

ret = DynPrec.DynTR(x0, func_single, {'single': 1}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory)

ret = DynPrec.DynTR(x0, func_double, {'double': 2}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory)

ret = DynPrec.DynTR(x0, func_dynamic, {'single': 1, 'double': 2}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory)

ret = scipy.optimize.minimize(func_bfgs, x0, method="L-BFGS-B", jac=True, bounds=None, options={'ftol': 0, 'gtol': eps, 'maxcor': max_memory, 'maxiter': maxit})

def test_all():


    eps = 1.0e-6
    epsTR = eps
    maxit = 1000
    max_memory = 10
    temp = pycutest.all_cached_problems()
    singleTR = list()
    doubleTR = list()
    lbfgs = list()
    dynTR = list()

    max_problem_dim = 101
    sing_file = 'single_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    doub_file = 'double_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    dyn_file = 'dynTR_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    bfgs_file = 'lbfgs_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'



    fields = ['problem', 'success', 'time', 'feval', 'gradnorm', 'nits', 'fevals', 'single_evals','message']

    problems = pycutest.find_problems(constraints='U') #, n=[1,5000])
    for (i, prob_str) in enumerate(problems):
        if 'JIMACK' not in prob_str and 'BA-' not in prob_str:
            if os.path.exists(cache_dir+prob_str+"_single"):

                # construct function handles for single, double, and dynamic precision
                p1 = pycutest.import_problem(prob_str+"_single")
                func_single = lambda z, prec: p1.obj((prec/prec)*z, gradient=True)

                p2 = pycutest.import_problem(prob_str+"_double")
                func_double = lambda z, prec: p2.obj((prec/prec)*z, gradient=True)

                func_dynamic = lambda z, prec: pycutest_wrapper(z, prec, p1, p2)
                func_bfgs = lambda z: p2.obj(z, gradient=True)

                if p1.n <= max_problem_dim:
                    print('\n')
                    print(i+1, '. Solving problem', prob_str, " in dim=", p1.n)
                    x0 = p1.x0

                    ### SINGLE PRECISION ###
                    print('Solving single TR.', end=' ')
                    t0 = time.time()
                    ret = DynPrec.DynTR(x0, func_single, [1], gtol=eps, max_iter=budget, tr_tol=epsTR, verbose=False, max_memory=max_memory)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:15] + ')'
                    print(mystr)
                    if ret.success:
                        success='converged'
                    else:
                        success='failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, ret.nfev, ret.message]
                    singleTR.append(temp)


                    ### DOUBLE PRECISION ###
                    t0 = time.time()
                    print('Solving double TR.', end=' ')
                    ret = DynPrec.DynTR(x0, func_double, [2], gtol=eps, max_iter=budget, tr_tol=epsTR, verbose=False, max_memory=max_memory)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:15] + ')'
                    print(mystr)
                    if ret.success:
                        success='converged'
                    else:
                        success='failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, 0, ret.message]
                    doubleTR.append(temp)


                    ### LBFGS SOLVER ###
                    t0 = time.time()
                    print('Solving LBFGS.', end=' ')
                    ret = scipy.optimize.minimize(func_bfgs, x0, method="L-BFGS-B", jac=True, bounds=None, options={'ftol': 0, 'gtol': eps, 'maxcor': max_memory, 'maxiter': maxit})
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:25] + ')'
                    print(mystr)
                    if ret.success:
                        if 'REL_RED' in ret.message:
                            success='failed'
                        else:
                            success='converged'
                    else:
                        success='failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, 0, ret.message]
                    lbfgs.append(temp)


                    ### DYNAMIC PRECISION ###
                    t0 = time.time()
                    print('Solving dynamic TR.', end=' ')
                    ret = DynPrec.DynTR(x0, func_dynamic, [1,2], gtol=eps, max_iter=budget, tr_tol=epsTR, verbose=False, max_memory=max_memory)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:15] + ')'
                    print(mystr)
                    if ret.success:
                        success='converged'
                    else:
                        success='failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, ret.precision_counts[0], ret.message]
                    dynTR.append(temp)

                    if np.mod(i+1, 10) == 0 or p1.n > 100:
                        sing_df = pd.DataFrame(data=singleTR, columns=fields)
                        doub_df = pd.DataFrame(data=doubleTR, columns=fields)
                        dyn_df  = pd.DataFrame(data=dynTR, columns=fields)
                        bfgs_df = pd.DataFrame(data=lbfgs, columns=fields)
                        sing_df.to_csv(sing_file)
                        doub_df.to_csv(doub_file)
                        dyn_df.to_csv(dyn_file)
                        bfgs_df.to_csv(bfgs_file)


                else:
                    print('\n',i+1, '. ' + prob_str + ' problem exceeds maximum number of dimensions')

            else:
                print("Can't find file " + prob_str)


    sing_df = pd.DataFrame(data=singleTR, columns=fields)
    doub_df = pd.DataFrame(data=doubleTR, columns=fields)
    dyn_df  = pd.DataFrame(data=dynTR, columns=fields)
    bfgs_df = pd.DataFrame(data=lbfgs, columns=fields)

    sing_df.to_csv(sing_file)
    doub_df.to_csv(doub_file)
    dyn_df.to_csv(dyn_file)
    bfgs_df.to_csv(bfgs_file)
    """
    # REWRITE TO SAVE AS CSV FILE VIA PANDAS!!!!!
    curr_list = singleTR
    f = open("single.table", "w")
    f.write('--- \nalgname: SinglePrecTR \n')
    f.write('success: converged \n')
    f.write('free_format: True \n---\n')
    #line = ['name', 'success?', 'time', 'obj_val', 'grad_norm', 'nit', 'nfevals', 'message']
    #f.write("# problem name, time elapsed, success?, # iterations, # f_evals, final objective value, norm of gradient, stopping message \n")#, final x \n")
    #f.write("# %10s %10s %10s %15s %15s %10s %10s \t %25s \n" % tuple(line))
    for line in curr_list:
        f.write("%10s, %10s, %10.5f, %15.6E, %15.6E, %10d, %10d,  \t %25s \n" % tuple(line))
        #f.write("%10s %10s %10.5f %15.6E %15.6E %10d %10d  \t %25s \n" % tuple(line))
    f.close()
    """



def main():


    test_all()

if __name__ == "__main__":
    main()

