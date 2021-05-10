# Ensure compatibility with Python 2
from __future__ import print_function
import sys
import numpy as np
import pycutest
import FixedPrec

sys.path.append('/Users/clancy/Documents/_research/vp_trust/pycutest_cache')

eps = 1.0e-6
budget = 100

#prob_str = "ARGTRIGLS"
prob_str = "ARWHEAD"
p = pycutest.import_problem(prob_str)

print(prob_str, " function in %gD" % p.n)


iters = 0

x0 = p.x0
fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
#f, g = p.obj(x, gradient=True)  # objective and gradient


#H = p.hess(x)  # Hessian

xsol = FixedPrec.TR_fixed(x0, fun, 1, 1, eps, budget)
#TR_fixed(x0, fun, hessian, delta0, eps, budget):


print(xsol)
"""
while iters < 100 and np.linalg.norm(g) > 1e-10:
    print("Iteration %g: objective value is %g with norm of gradient %g at x = %s" % (iters, f, np.linalg.norm(g), str(x)))


    s = np.linalg.solve(H, -g)  # Newton step
    x = x + s  # used fixed step length
    f, g = p.obj(x, gradient=True)


    H = p.hess(x)
    iters += 1
"""

#print("Found minimum x = %s after %g iterations" % (str(xsol), iters))
print("Done")
