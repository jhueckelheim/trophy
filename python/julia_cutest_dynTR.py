import sys
import os
import numpy as np
import julia
from julia import Main
import tensorflow as tf
import time
from DynPrec_edits import DynTR

"""
1. Create function handle that calls Julia with variable and precision as the inputs. 
2. Establish initializations
3. Feed function into Dynamic Trust Region solver. 
4. Return and prints details. 

"""


def objective_func(x, precision, problem, julia_main):
    if precision == 'half':
        typ = 'float16'
        julia_main.x = np.float16(x)
    if precision == 'single':
        typ = 'float32'
        julia_main.x = np.float32(x)
    if precision == 'double':
        typ = 'float64'
        julia_main.x = np.float64(x)
    f, grad = julia_main.eval(problem + "(x)")
    f = eval('np.' + typ + '(f)')
    return f, grad


precision_dict = {'single': 1, 'double': 2}
#precision_dict = {'single': 1}
#precision_dict = {'double': 2}
np.random.seed(1)
prob_number = 1
function_path = "/Users/clancy/repos/trophy/Julia/functions/cutest/"
problem_names = ['fletchcr', 'genrose', 'power']
curr_prob = problem_names[prob_number]
file = function_path + curr_prob + '_python.jl'

file = function_path + curr_prob + '_python.jl'
julia.Main.include(file)

fun = lambda z, prec: objective_func(z, prec, curr_prob, julia.Main)

x0 = np.random.normal(0, 1, (100,))

ret = DynTR(x0, fun, precision_dict, gtol=1.0e-5, max_iter=1000, verbose=True, max_memory=10, store_history=False,
          tr_tol=1e-6, delta_init=None, max_delta=1e4, sr1_tol=1.e-4, write_folder=None)



"""
#Main.include(function_path + curr_prob + "_python.jl")

n_sims = 20
julia_call = curr_prob + "(x)"
total_time = 0
ti = time.time()
for i in range(n_sims):
    x = np.random.normal(0, 1, (10000,))
    Main.x = np.float32(x)
    #Main.x = np.array(x, dtype=tf.bfloat16)
    ti = time.time()
    sum, grad = Main.eval(julia_call)
    print(sum)
    tf = time.time()
    total_time += tf-ti


print('Average time per evaluation', total_time/n_sims)
"""