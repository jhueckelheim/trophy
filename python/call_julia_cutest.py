import sys
import os
import numpy as np
import julia
from julia import Main
import tensorflow as tf
import time

"""
1. Create function handle that calls Julia with variable and precision as the inputs. 
2. Establish initializations
3. Feed function into Dynamic Trust Region solver. 
4. Return and prints details. 

"""
np.random.seed(1)
prob_number = 2
function_path = "/Users/clancy/repos/trophy/Julia/functions/cutest/"
problem_names = ['fletchcr', 'genrose', 'power']
curr_prob = problem_names[prob_number]
file = function_path + curr_prob + '_python.jl'



# initialize the starting vector for fletchcr, this should accept any size vector



Main.include(function_path + curr_prob + "_python.jl")

n_sims = 1
julia_call = curr_prob + "(x)"
total_time = 0
ti = time.time()
for i in range(n_sims):
    x = np.random.normal(0, 1, (10,))
    Main.x = np.float32(x)
    #Main.x = np.array(x, dtype=tf.bfloat16)
    ti = time.time()
    sum, grad = Main.eval(julia_call)
    print(grad.dtype)
    tf = time.time()
    total_time += tf-ti


print('Average time per evaluation', total_time/n_sims)
