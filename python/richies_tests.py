# Ensure compatibility with Python 2

import numpy as np
import pycutest
import FixedPrec
import sys, os, time

# add pycutestcache path if you haven't done so elsewhere
sys.path.append('/Users/clancy/Documents/_research/vp_trust/pycutest_cache')

def rosen_test():
    p = pycutest.import_problem('ROSENBR')
    #p = pycutest.import_problem('SINEVAL')

    print("Rosenbrock function in %gD" % p.n)

    iters = 0

    x = p.x0
    f, g = p.obj(x, gradient=True)  # objective and gradient
    print(g)
    H = p.hess(x)  # Hessian

    while iters < 100 and np.linalg.norm(g) > 1e-10:
        print("Iteration %g: objective value is %g with norm of gradient %g at x = %s" % (iters, f, np.linalg.norm(g), str(x)))
        s = np.linalg.solve(H, -g)  # Newton step
        x = x + s  # used fixed step length
        f, g = p.obj(x, gradient=True)
        H = p.hess(x)
        iters += 1

    print("Found minimum x = %s after %g iterations" % (str(x), iters))
    print("Done")





def test_all():
    eps = 1.0e-6
    budget = 1000
    temp = pycutest.all_cached_problems()
    problems1 = []
    for p in temp:
        problems1.append(p[0])
    problems = ['JUDGE', 'FBRAIN3LS', 'WAYSEA2', 'BROWNDEN', 'HILBERTA', 'PALMER5D', 'BOXBODLS', 'HIMMELBB', 'ENGVAL2', 'ENSOLS', 'PRICE4', 'CERI651ELS', 'SISSER', 'TRIGON1', 'S308NE', 'GROWTHLS', 'SINEVAL', 'GAUSS2LS', 'STRATEC', 'NELSONLS', 'MISRA1ALS', 'WAYSEA1', 'GAUSS3LS', 'VESUVIALS', 'HAIRY', 'YFITU', 'HIMMELBCLS', 'MARATOSB', 'LSC2LS', 'PALMER1C', 'POWELLBSLS', 'HIMMELBG', 'DENSCHNF', 'COOLHANSLS', 'PRICE3', 'VESUVIOULS', 'KOWOSB', 'HIMMELBH', 'ZANGWIL2', 'PALMER2C', 'HEART8LS', 'HATFLDD', 'MISRA1DLS', 'BRKMCC', 'PALMER8C', 'DENSCHNA', 'MISRA1BLS', 'DENSCHNC', 'S308', 'SNAIL', 'EXPFIT', 'BOX3', 'ECKERLE4LS', 'HAHN1LS', 'MGH17LS', 'PALMER1D', 'HIMMELBF', 'JENSMP', 'CERI651DLS', 'GBRAINLS', 'CHWIRUT2LS', 'DEVGLA2NE', 'ALLINITU', 'VESUVIOLS', 'THURBERLS', 'CERI651ALS', 'VIBRBEAM', 'GAUSS1LS', 'DJTL', 'LSC1LS', 'PALMER6C', 'CERI651CLS', 'PALMER5C', 'OSBORNEA', 'BIGGS6', 'PALMER7C', 'BEALE', 'MGH10SLS', 'RAT43LS', 'CLUSTERLS', 'HELIX', 'MEXHAT', 'DENSCHNB', 'MGH09LS', 'MEYER3', 'GULF', 'DENSCHND', 'BROWNBS', 'SSI', 'HATFLDE', 'KIRBY2LS', 'BENNETT5LS', 'DANWOODLS', 'DANIWOODLS', 'RAT42LS', 'DEVGLA1', 'HATFLDFL', 'STREG', 'AKIVA', 'LANCZOS2LS', 'ROSENBR', 'HATFLDFLS', 'HIELOW', 'HUMPS', 'BARD', 'MGH10LS', 'DEVGLA2', 'CHWIRUT1LS', 'EGGCRATE', 'HILBERTB', 'RECIPELS', 'LANCZOS1LS', 'PALMER3C', 'ELATVIDU', 'ROSZMAN1LS', 'POWERSUM', 'ROSENBRTU', 'GAUSSIAN', 'PALMER4C', 'MISRA1CLS', 'DENSCHNE', 'CLIFF', 'STRTCHDV', 'TRIGON2', 'HEART6LS', 'POWELLSQLS', 'EXP2', 'LANCZOS3LS', 'LOGHAIRY', 'CERI651BLS', 'CUBE']
    #problems = set(problems1).intersection(problems2)
    #bad_problems = ["MODBEALE", "DEVGLA1", "ENGVAL2", "POWER"]
    for (i, prob_str) in enumerate(problems):
        #prob_str = prob[0]
        if prob_str: # not in bad_problems:
            #prob_str = prob[0]
            p = pycutest.import_problem(prob_str)
            print(i, '. Solving problem', prob_str, " in dim=", p.n)
            x0 = p.x0
            fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
            #try:
            xsol = FixedPrec.TR_fixed(x0, fun, 1, 1, eps, budget)
            if xsol is not None:
                fsol, gsol = fun(xsol)
                print("Done with ", prob_str, " with ||g|| = ", np.linalg.norm(gsol))
            else:
                print('Failed to solve problem in max iterations')
            #except:

            #    print("Solution not found in max number of iterations.")
            time.sleep(2)
        else:
            print("Skipping ", prob_str)
            time.sleep(2)



def testFixedPrec():
    eps = 1.0e-6
    budget = 1000

    prob_str = "ARGTRIGLS"
    #
    #prob_str = "ARWHEAD"
    #prob_str = "SROSENBR"
    #prob_str = "JUDGE"
    #prob_str = "FBRAIN3LS"
    #prob_str = "WAYSEA2"
    #prob_str = "BROWNDEN"
    #prob_str = "HILBERTA"
    #prob_str = "PALMER5D"
    #prob_str = "BOXBODLS"
    p = pycutest.import_problem(prob_str)

    print(prob_str, " function in %gD" % p.n)


    iters = 0

    x0 = p.x0
    fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
    #f, g = p.obj(x, gradient=True)  # objective and gradient


    #H = p.hess(x)  # Hessian
    #try:
    xsol = FixedPrec.TR_fixed(x0, fun, 1, 1, eps, budget)
    f,g = fun(xsol)
    print(xsol)
    #except:
    #    print("Didn't converge")
    #TR_fixed(x0, fun, hessian, delta0, eps, budget):





    #print("Found minimum x = %s after %g iterations" % (str(xsol), iters))
    print("Done")


def main():

    test_all()
    #testFixedPrec()



if __name__ == "__main__":
    main()

