from scipy import optimize
import numpy as np
from numpy.linalg import norm
from utilimport get_cartesian_coords, get_spherical_coords

"""
TO DO: Several places where indices may need to be corrected and must make changes for calls to TRS
"""


def DynTR(x0, handle, prec_vec, epsilon, budget):
    """
    :param x0: numpy array, initialization
    :param handle:  objective/gradient handle
    :param prec_vec: vector of precision values
    :param epsilon: final gradient accuracy
    :param budget: budget constraint
    :return:
    """
    # Later: Check that prec_vec is listed in ascending order
    num_prec = length(prec_vec)
    prec_lvl = 1                        # MIGHT NEED TO ADJUST INDEXING
    prec = prec_vec[prec_lvl]

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
    print("Function evaluations at precision level ", prec, " bits:")
    f,g = handle(x,prec)

    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    max_delta = 1.0e3
    delta = 1.0
    # initial print statement:
    print("iter: 0, f: ",f, ", delta: ", delta, ", norm(g): ", norm(g))

    # Initialize Hessian - max_memory should be exposed
    max_memory = 30
    memory = min(max_memory,n+1)
    Y = 0.0 #Float64[]                          # come back to this a little later
    S = 0.0 #Float64[]                           # come back to this a little later
    B0 = np.eye(n)
    H = np.zeros((n,n))

    # benchmarking:
    xhist = zeros((budget+1,n), dtype='np.float64')
    prechist = str()

    iter = 1
    first_fail = True
    first_success = 0
    theta = 0.0


    # this is the main loop that is iterate through for TROPHY
    for i in range(budget):   # note that this indexes from 0 and not 1, check for errors because of it
        # benchmarking:
        prec_str = ""

        # criticality check
        # We impose stopping criteria here. If grad is small, stop. If the trust region radius has shrunken
        # considerably, then we increase precision I guess...seems like this could cause problems elsewhere.
        if (norm(g) <= epsilon) or (delta <= np.sqrt(machine_eps)):
            if prec_lvl == num_prec:
                # terminate, because we're done
                return prechist, xhist
            else:
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                print("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f,g = handle(x,prec)
                # hard-coded benchmarking:
                if prec_lvl == 1:
                    prec_str = prec_str + "s"
                elif prec_lvl == 2:
                    prec_str = prec_str + "d"

        # solve TR subproblem
        # H can become slightly asymmetric and then TRS will complain. so:
        # If we haven't succeeded yet, then
        if first_success == 0:
            # Do we just pass the Zero matrix initially?
            #CG_Steinhaug_matFree(epsTR, g , deltak, S,Y,nv):
            sol = trs(H,g,delta)                    # need an appropriate call to some trust region solver
        else:
            H = (H+H.T)/2.0
            try:
                # instead of providing matrix, only use matrices Y and S
                sol = trs(H,g,delta)             # need an appropriate call to some trust region solver
            except:
                sol = trs(0*B0,g,delta)          # need an appropriate call to some trust region solver



        s = sol[1]          # we will likely need to change the indices here for a different solver
        s = s[:,1]

        # evaluate the surrogate. Can we evaluate without forming the Hessian?
        val = 0.5*(s@H)@s + s@g

        # evaluate function at current precision at new trial point
        fplus,gplus = handle(x+s,prec)
        # hard-coded benchmarking:
        if prec_lvl == 1:
            prec_str = prec_str + "s"
        elif prec_lvl == 2:
            prec_str = prec_str + "d"

        # determine acceptance of trial point
        ared = f - fplus
        rho = ared/(-1.0*val)          # val = m(s) - m(0) but we want m(0)-m(s)

        # for Hessian updating:
        gprev = g

        # failed to reduce by desired amount (or actually increase)
        if (rho < eta_good) or (val > 0):
            # the iteration was a failure
            xnew = x  # don't advance in s direction
            # do we update the precision?
            # everything within this if only occurs the first time a TR failure occurs.
            if first_fail:
                first_fail = False
                if prec_lvl < num_prec:
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[prec_lvl + 1]   # seems like in algo, this should be full precision, i.e., double
                    print("Probed a pair of function evaluations at precision level ", temp_prec, " bits:")
                    ftemp,gtemp = handle(x,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1:
                        prec_str = string(prec_str, "s")
                    elif prec_lvl + 1 == 2:
                        prec_str = prec_str + "d"
                    ftempplus,gtempplus = handle(x+s,temp_prec)   # evaluate in next higher precision...not necessarily double
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1:
                        prec_str = prec_str + "s"
                    elif prec_lvl + 1 == 2:
                        prec_str = string(prec_str, "d")
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if isnan(theta):
                        theta = sqrt(machine_eps)

                    if (theta > eta_good*(-g@s - 0.5*(s@H)@s)) and (delta < min(1.0,norm(gtemp))):
                        # update precision
                        prec_lvl += 1
                        print("Permanently switching to precision level ", temp_prec, " bits:")
                        f = ftemp
                        g = gtemp
                        gplus = gtemp
                        delta = min(norm(g),1.0)

                        xnew = x + s
                        gprev = gtemp
                        # testing:
                        #Y = Float64[]
                        #S = Float64[]
                    else:
                        # just shrink the TR
                        delta = gamma_dec*norm(s)

                else: # we can't increase the precision anymore, so just reject
                    delta = gamma_dec*norm(s)

            else:
                if (theta > eta_good*(-(g@s) - 0.5*(s@H)@s)) and (prec_lvl < num_prec) and (delta < min(1.0,norm(g))):
                    temp_prec = prec_vec[prec_lvl + 1]
                    print("Probed a pair of function evaluations at ", temp_prec, " bits...")
                    ftemp, gtemp = handle(x,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1:
                        prec_str = prec_str + "s"
                    elif prec_lvl + 1 == 2:
                        prec_str = prec_str + "d"

                    ftempplus,gtempplus = handle(x+s,temp_prec)   #  I think this should be full precision
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1:
                        prec_str = prec_str + "s"
                    elif prec_lvl + 1 == 2:
                        prec_str = prec_str + "d"

                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if theta > eta_good*(-g@s - 0.5(s@H)@s):
                        prec_lvl += 1
                        print("Permanently switching to precision level ", temp_prec, " bits:")
                        f = ftemp
                        g = gtemp
                        fplus = ftempplus
                        gplus = gtempplus
                        gprev = gtemp
                        delta = min(1.0,norm(g))
                        # testing:
                        #Y = []
                        #S = []

                else: # the model quality isn't suspect, standard reject
                    delta = gamma_dec*delta

        # the step is at least good, maybe great
        else:
            if first_success == 0:
                first_success = 1

            xnew = x + s
            f = fplus
            g = gplus
            # is the step great?  # Maybe we sho
            if (norm(s) > 0.8*delta) and (rho > eta_great):
                delta = min(max_delta,gamma_inc*delta)
            else:
                delta = norm(s)

        # upon first success, store first change in gradient set approximate Hessian to
        if first_success == 1:
            y0 = gplus-gprev
            #constant = (y0'*s)/(y0'*y0)
            constant = 1.0
            B0 = constant*B0
            H = B0
            first_success = 2

        # update the Hessian
        if first_success > 0:
            H, Y, S = update_hessian(H,B0,Y,S,gplus,gprev,s,memory,sr1_tol)




        # get ready for next iteration
        x = xnew

        # TODO: write better output to command line indicating progress
        print("iter: ", iter, ", f: ",f, ", delta: ", delta, ", norm(g): ", norm(g))
        if isnan(f) and isnan(delta):
            break

        # early termination if progress seems unlikely
        if norm(g) < epsilon:
            if prec_lvl == num_prec:
                return (prechist, xhist)
            #else
            #    prec_lvl += 1
            #    prec = prec_vec[prec_lvl]
            #    println("Function evaluations at precision level ", prec_vec[prec_lvl], " bits:")
            #    f,g = handle(x,prec)


        iter += 1
        # end of iteration benchmarking:
        #push!(prechist,prec_str)
        prechist.append(prec_str)
        xhist[iter,:] = x    # since x is a numpy array, don't need to transpose
        # end main for loop
        return prechist, xhist
# end DynTR


#function update_hessian(H::Matrix,B0::Matrix,Y::Array,S::Array,gplus::Array,gprev::Array,s::Array,memory::Integer,sr1_tol::Float64)
def update_hessian(H, B0, Y, S, gplus, gprev, s, memory, sr1_tol):

    y = gplus-gprev
    pred_grad = y - H@s
    dot_prod = pred_grad@s
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):
        if bool(Y):
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]


        # add the newest (y,s) pair
        if not bool(Y):
            Y = y
            S = s
        else:
            Y = np.hstack((Y,y))
            S = np.hstack((S,s))


        Psi = Y-B0@S
        SY = S.T@Y

        #if typeof(SY)!=Float64
        if SY.shape[0] != 1:   # check to see if matrix
            D = np.diag(np.diag(SY))
            L = np.tril(SY) - D
            U = np.triu(SY) - D
            try:
                M = np.linalg.inv( D+L+L.T - (S.T@B0)@S )
            except:
                n = length(y)
                inv(D+L+L.T - (S.T@B0)@S + sr1_tol*np.eye(Y.shape[1]))  #Matrix{Float64}(I,size(Y,2),size(Y,2)))
        else: # if not a matrix "invert" by dividing
            M = 1.0/(SY - (S.T@B0)@S)

        H = real(B0 + (Psi@M)@Psi.T)

    return H, Y, S

