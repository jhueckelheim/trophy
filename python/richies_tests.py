# Ensure compatibility with Python 2

import numpy as np
import pycutest
import FixedPrec
import DynPrec
import sys, os, time
from util_func_v2 import pycutest_wrapper

# add pycutestcache path if you haven't done so elsewhere
sys.path.append('/Users/clancy/Documents/_research/vp_trust/pycutest_cache')


def test_all():
    eps = 1.0e-6
    budget = 1000
    temp = pycutest.all_cached_problems()
    problems1 = []
    for p in temp:
        problems1.append(p[0])
    problems = ['JUDGE', 'FBRAIN3LS', 'WAYSEA2', 'BROWNDEN', 'HILBERTA', 'PALMER5D', 'BOXBODLS', 'HIMMELBB', 'ENGVAL2', 'ENSOLS', 'PRICE4', 'CERI651ELS', 'SISSER', 'TRIGON1', 'S308NE', 'GROWTHLS', 'SINEVAL', 'GAUSS2LS', 'STRATEC', 'NELSONLS', 'MISRA1ALS', 'WAYSEA1', 'GAUSS3LS', 'VESUVIALS', 'HAIRY', 'YFITU', 'HIMMELBCLS', 'MARATOSB', 'LSC2LS', 'PALMER1C', 'POWELLBSLS', 'HIMMELBG', 'DENSCHNF', 'COOLHANSLS', 'PRICE3', 'VESUVIOULS', 'KOWOSB', 'HIMMELBH', 'ZANGWIL2', 'PALMER2C', 'HEART8LS', 'HATFLDD', 'MISRA1DLS', 'BRKMCC', 'PALMER8C', 'DENSCHNA', 'MISRA1BLS', 'DENSCHNC', 'S308', 'SNAIL', 'EXPFIT', 'BOX3', 'ECKERLE4LS', 'HAHN1LS', 'MGH17LS', 'PALMER1D', 'HIMMELBF', 'JENSMP', 'CERI651DLS', 'GBRAINLS', 'CHWIRUT2LS', 'DEVGLA2NE', 'ALLINITU', 'VESUVIOLS', 'THURBERLS', 'CERI651ALS', 'VIBRBEAM', 'GAUSS1LS', 'DJTL', 'LSC1LS', 'PALMER6C', 'CERI651CLS', 'PALMER5C', 'OSBORNEA', 'BIGGS6', 'PALMER7C', 'BEALE', 'MGH10SLS', 'RAT43LS', 'CLUSTERLS', 'HELIX', 'MEXHAT', 'DENSCHNB', 'MGH09LS', 'MEYER3', 'GULF', 'DENSCHND', 'BROWNBS', 'SSI', 'HATFLDE', 'KIRBY2LS', 'BENNETT5LS', 'DANWOODLS', 'DANIWOODLS', 'RAT42LS', 'DEVGLA1', 'HATFLDFL', 'STREG', 'AKIVA', 'LANCZOS2LS', 'ROSENBR', 'HATFLDFLS', 'HIELOW', 'HUMPS', 'BARD', 'MGH10LS', 'DEVGLA2', 'CHWIRUT1LS', 'EGGCRATE', 'HILBERTB', 'RECIPELS', 'LANCZOS1LS', 'PALMER3C', 'ELATVIDU', 'ROSZMAN1LS', 'POWERSUM', 'ROSENBRTU', 'GAUSSIAN', 'PALMER4C', 'MISRA1CLS', 'DENSCHNE', 'CLIFF', 'STRTCHDV', 'TRIGON2', 'HEART6LS', 'POWELLSQLS', 'EXP2', 'LANCZOS3LS', 'LOGHAIRY', 'CERI651BLS', 'CUBE']
    #problems = set(problems1).intersection(problems2)
    #bad_problems = CERI651ELS
    for (i, prob_str) in enumerate(problems):
        if prob_str:
            #p = pycutest.import_problem(prob_str, precision="single")
            p = pycutest.import_problem(prob_str + "single")
            print(i, '. Solving problem', prob_str, " in dim=", p.n)
            x0 = p.x0
            fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
            #try:
            xsol = FixedPrec.TR_fixed(x0, fun, eps, budget)
            if xsol is not None:
                fsol, gsol = fun(xsol)
                print("Done with ", prob_str, " with ||g|| = ", np.linalg.norm(gsol))
            else:
                print('Failed to solve problem in max iterations. Done with ', prob_str, ', dimension', p.n)

            time.sleep(2)
        else:
            print("Skipping ", prob_str)
            time.sleep(2)

def testFixedPrec():
    eps = 1.0e-8
    budget = 1000

    #prob_str = "ARGTRIGLS"
    #
    #prob_str = "ARWHEAD"
    #prob_str = "SROSENBR"
    #prob_str = "SROSENBR_sp"
    #prob_str = "JUDGE"
    #prob_str = "FBRAIN3LS"
    #prob_str = "WAYSEA2"
    #prob_str = "BROWNDEN"
    #prob_str = "HILBERTA"
    #prob_str = "PALMER5D"
    #prob_str = "BOXBODLS"
    #prob_str = "HIMMELBB"
    prob_str = "CERI651ELS"
    #prob_str = "SISSER"
    #prob_str = "ENSOLS"
    #pycutest.find_problems()
    #prob_str = "AKIVA"

    # Set the -sp flag in sifOptions to compile single precision code, i.e. p = pycutest.import_problem(prob_str, "single", None, None, ["-sp"])
    p = pycutest.import_problem(prob_str, precision="double")#

    print(prob_str, " function in %gD" % p.n)


    iters = 0

    x0 = p.x0
    fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
    #H = p.hess(x)  # Hessian
    #try:
    xsol = FixedPrec.TR_fixed(x0, fun, eps, budget)
    f,g = fun(xsol)
    print(xsol)
    print("Done \n \n")




def testDynPrec():
    eps = 1.0e-8
    budget = 1000
    #prob_str = "ARGTRIGLS"
    prob_str = "ARWHEAD"
    #prob_str = "ROSENBR"
    #prob_str = "SISSER"
    prob_str = "MISRA1BLS"

    #prob_str = "AKIVA_sp"
    #p1 = pycutest.import_problem(prob_str, precision="single")
    #p1 = pycutest.import_problem(prob_str, precision="single")
    p1 = pycutest.import_problem(prob_str+"_single", precision="single")
    p2 = pycutest.import_problem(prob_str+"_double", precision="double")
    func = lambda z, prec: pycutest_wrapper(z, prec, p1, p2)

    """
    for i in range(100):
        x = np.random.normal(0,1,p1.x0.shape)
        f1 = func(x, 1)
        f2 = func(x, 2)
        print("\nSingle eval=",f1[0])
        print("Double eval=",f2[0])
        print('Eval diff=', abs(f1[0]-f2[0]))
        #print('Grad diff', np.linalg.norm(f1[1]-f2[1]))
    """

    print(prob_str, " function in %gD" % p1.n)

    x0 = p1.x0
    prec_vec = [1,2]
    precision_history, solution_history = DynPrec.DynTR(x0, func, prec_vec, gtol=eps, max_iter=budget, verbose = True)
    xsol = solution_history[-1]
    print(xsol)



def test_all_Dyn():
    eps = 1.0e-6
    budget = 1000
    temp = pycutest.all_cached_problems()
    #for p in temp:
    #    problems1.append(p[0])
    problems = ['JUDGE', 'FBRAIN3LS', 'WAYSEA2', 'BROWNDEN', 'HILBERTA', 'PALMER5D', 'BOXBODLS', 'HIMMELBB', 'ENGVAL2', 'ENSOLS', 'PRICE4', 'CERI651ELS', 'SISSER', 'TRIGON1', 'S308NE', 'GROWTHLS', 'SINEVAL', 'GAUSS2LS', 'STRATEC', 'NELSONLS', 'MISRA1ALS', 'WAYSEA1', 'GAUSS3LS', 'VESUVIALS', 'HAIRY', 'YFITU', 'HIMMELBCLS', 'MARATOSB', 'LSC2LS', 'PALMER1C', 'POWELLBSLS', 'HIMMELBG', 'DENSCHNF', 'COOLHANSLS', 'PRICE3', 'VESUVIOULS', 'KOWOSB', 'HIMMELBH', 'ZANGWIL2', 'PALMER2C', 'HEART8LS', 'HATFLDD', 'MISRA1DLS', 'BRKMCC', 'PALMER8C', 'DENSCHNA', 'MISRA1BLS', 'DENSCHNC', 'S308', 'SNAIL', 'EXPFIT', 'BOX3', 'ECKERLE4LS', 'HAHN1LS', 'MGH17LS', 'PALMER1D', 'HIMMELBF', 'JENSMP', 'CERI651DLS', 'GBRAINLS', 'CHWIRUT2LS', 'DEVGLA2NE', 'ALLINITU', 'VESUVIOLS', 'THURBERLS', 'CERI651ALS', 'VIBRBEAM', 'GAUSS1LS', 'DJTL', 'LSC1LS', 'PALMER6C', 'CERI651CLS', 'PALMER5C', 'OSBORNEA', 'BIGGS6', 'PALMER7C', 'BEALE', 'MGH10SLS', 'RAT43LS', 'CLUSTERLS', 'HELIX', 'MEXHAT', 'DENSCHNB', 'MGH09LS', 'MEYER3', 'GULF', 'DENSCHND', 'BROWNBS', 'SSI', 'HATFLDE', 'KIRBY2LS', 'BENNETT5LS', 'DANWOODLS', 'DANIWOODLS', 'RAT42LS', 'DEVGLA1', 'HATFLDFL', 'STREG', 'AKIVA', 'LANCZOS2LS', 'ROSENBR', 'HATFLDFLS', 'HIELOW', 'HUMPS', 'BARD', 'MGH10LS', 'DEVGLA2', 'CHWIRUT1LS', 'EGGCRATE', 'HILBERTB', 'RECIPELS', 'LANCZOS1LS', 'PALMER3C', 'ELATVIDU', 'ROSZMAN1LS', 'POWERSUM', 'ROSENBRTU', 'GAUSSIAN', 'PALMER4C', 'MISRA1CLS', 'DENSCHNE', 'CLIFF', 'STRTCHDV', 'TRIGON2', 'HEART6LS', 'POWELLSQLS', 'EXP2', 'LANCZOS3LS', 'LOGHAIRY', 'CERI651BLS', 'CUBE']
    #problems = set(problems1).intersection(problems2)
    #bad_problems = CERI651ELS

    for (i, prob_str) in enumerate(problems):
        if prob_str:
            #p = pycutest.import_problem(prob_str, precision="single")
            #print(i, '. Solving problem', prob_str, " in dim=", p.n)

            #fun = lambda z: p.obj(z, gradient=True)  # objective and gradient
            p1 = pycutest.import_problem(prob_str+"_single", precision="single")
            p2 = pycutest.import_problem(prob_str+"_double", precision="double")
            x0 = p1.x0
            fun = lambda z, prec: pycutest_wrapper(z, prec, p1, p2)
            print(i, '. Solving problem', prob_str, " in dim=", p1.n)
            if p1.n > 10:
                print('Why was this included')

            prec_vec = [1,2]
            prechist, xsol_array = DynPrec.DynTR(x0, fun, prec_vec, gtol=eps, max_iter=budget)
            try:
                xsol = xsol_array[-1]
                fsol, gsol = fun(xsol, 2)
                print("Done with ", prob_str, " with ||g|| = ", np.linalg.norm(gsol))
            except:
                print("No solution returned")
            if xsol is None:
                print('Failed to solve problem in max iterations. Done with ', prob_str, ', dimension', p1.n)

            #time.sleep(1)
        else:
            print("Skipping ", prob_str)
            time.sleep(2)



def test_one_precision_level():
    eps = 1.0e-6
    budget = 1000
    prec_vec = [1]
    prob_str = 'WAYSEA2'
    p = pycutest.import_problem(prob_str+"_double")#, precision="single")
    x0 = p.x0
    fun = lambda z, lvl: p.obj(z, gradient=True)  # objective and gradient
    #def temp_fun(z,lvl):


    ret = DynPrec.DynTR(x0, fun, prec_vec, gtol=eps, max_iter=budget, verbose=True)





def main():


    #test_all()
    #testFixedPrec()
    #testDynPrec()
    #test_all_Dyn()
    test_one_precision_level()

if __name__ == "__main__":
    main()

