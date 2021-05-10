# Example: using problem classification system
import sys
import pycutest

# keeps overwriting my Python path...just add here
sys.path.append('/Users/clancy/Documents/_research/vp_trust/pycutest_cache')


# Find unconstrained, variable-dimension problems
probs = pycutest.find_problems(constraints='U', userN=True)
print(probs)

# Properties of problem ROSENBR
#print(pycutest.problem_properties('ROSENBR'))





# Print parameters for problem ARGLALE
#pycutest.print_available_sif_params('ROSENBR')
pycutest.print_available_sif_params('ARGLINA')

# Build this problem with N=100, M=200 'ARGLALE'
problem = pycutest.import_problem('ROSENBR')#, sifParams={'N':100, 'M':200})
problem = pycutest.import_problem('ARGLINA')#, sifParams={'N':100, 'M':200})
print(problem)



