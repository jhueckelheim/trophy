import os

import numpy as np
import pycutest
import sys, os

sys.path.append('/Users/clancy/Documents/_research/vp_trust/pycutest_cache')
probs = pycutest.find_problems(constraints='U')# , n=[2,10])

bad_problems = ['BA-L16LS', 'BA-L1LS', 'JIMACK']#, 'BA-L73LS']

bad_problems_new = list()
probs = "TRIDIA"
os.chdir('/Users/clancy/Documents/_research/vp_trust/pycutest_cache/pycutest_cache_holder')
for (i, problem) in enumerate(probs):
    if problem not in bad_problems and '-' not in problem:
        print(i,'. Importing problem', problem)
        # initially, all problems will be empty, so run once, preduce, relabel folder, then run again
        if not (os.path.exists(problem+"_single")):
            p = pycutest.import_problem(problem, precision="single")
            cmd = 'mv ' + problem + ' ' + problem + '_single'
            os.system(cmd)
        else: print(problem+"_single already cached")
        if not (os.path.exists(problem+"_double")):
            p = pycutest.import_problem(problem, precision="double")
            cmd = 'mv ' + problem + ' ' + problem + '_double'
            os.system(cmd)
        else: print(problem+"_double already cached")
    else:
        print(problem, "is broken, don't import")
        pause(5)
        bad_problems_new.append(problem)