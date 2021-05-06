module DynPrec

using TRS, LinearAlgebra

export DynTR

function DynTR(x0::Array{Float64,1},handle,prec_vec::Array{Int64,1},epsilon::Float64,budget::Int64)

    # Later: Check that prec_vec is listed in ascending order
    num_prec = length(prec_vec)
    prec_lvl = 1
    prec = prec_vec[prec_lvl]

    # Set the initial iterate
    x = x0
    n = length(x)

    # Internal algorithmic parameters: Should expose through a struct Type
    eta_good = 0.01
    eta_great = 0.1
    gamma_dec = 0.5
    gamma_inc = 2
    sr1_tol = 1e-4

    # First function evaluation:
    println("Function evaluations at precision level ", prec, " bits:")
    f,g = handle(x,prec)

    # Initial TR radius - max_delta, delta could be exposed as an initial parameter
    max_delta = 1e3
    delta = 1.0
    # initial print statement:
    println("iter: 0, f: ",f, ", delta: ", delta, ", norm(g): ", norm(g))

    # Initialize Hessian - max_memory should be exposed
    max_memory = 30
    memory = min(max_memory,n+1)
    #println("memory = ",memory," n = ",n)
    Y = Float64[]
    S = Float64[]
    B0 = Matrix{Float64}(I,n,n)
    H = zeros(Float64,n,n)

    # benchmarking:
    xhist = zeros(Float64,budget+1,n)
    prechist = String[]

    iter = 1
    first_fail = true
    first_success = 0
    theta = 0.0
    for i = 1:budget
        # benchmarking:
        prec_str = ""

        # criticality check
        if norm(g) <= epsilon || delta <= sqrt(eps())
            if prec_lvl == num_prec
                # terminate, because we're done
                return prechist, xhist
            else
                prec_lvl += 1
                prec = prec_vec[prec_lvl]
                println("Permanently switching evaluations to precision level ", prec_vec[prec_lvl], " bits:")
                f,g = handle(x,prec)
                # hard-coded benchmarking:
                if prec_lvl == 1
                    prec_str = string(prec_str, "s")
                elseif prec_lvl == 2
                    prec_str = string(prec_str, "d")
                end
            end
        end

        # solve TR subproblem
        # H can become slightly asymmetric and then TRS will complain. so:

        if first_success == 0
            sol = trs(H,g,delta)
        else
            H = (H+H')/2.0
            sol = try
                trs(H,g,delta)
            catch err
                trs(0*B0,g,delta)
            end
        end

        s = sol[1]
        s = s[:,1]
        val = 0.5*s'*H*s + s'*g

        # evaluate function at current precision at new trial point
        fplus,gplus = handle(x+s,prec)
        # hard-coded benchmarking:
        if prec_lvl == 1
            prec_str = string(prec_str, "s")
        elseif prec_lvl == 2
            prec_str = string(prec_str, "d")
        end

        # determine acceptance of trial point
        ared = f - fplus
        rho = ared/(-1.0*val)

        # for Hessian updating:
        gprev = g

        if rho < eta_good || val > 0
            # the iteration was a failure
            xnew = x
            # do we update the precision?
            if first_fail
                first_fail = false
                if prec_lvl < num_prec
                    # evaluate x, x+s at next highest precision
                    temp_prec = prec_vec[prec_lvl + 1]
                    println("Probed a pair of function evaluations at precision level ", temp_prec, " bits:")
                    ftemp,gtemp = handle(x,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1
                        prec_str = string(prec_str, "s")
                    elseif prec_lvl + 1 == 2
                        prec_str = string(prec_str, "d")
                    end
                    ftempplus,gtempplus = handle(x+s,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1
                        prec_str = string(prec_str, "s")
                    elseif prec_lvl + 1 == 2
                        prec_str = string(prec_str, "d")
                    end
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if isnan(theta)
                        theta = sqrt(eps())
                    end

                    if theta > eta_good*(-g'*s - 0.5*s'*H*s) && delta < min(1.0,norm(gtemp))
                        # update precision
                        prec_lvl += 1
                        println("Permanently switching to precision level ", temp_prec, " bits:")
                        f = ftemp
                        g = gtemp
                        gplus = gtemp
                        delta = min(norm(g),1.0)

                        xnew = x + s
                        gprev = gtemp
                        # testing:
                        #Y = Float64[]
                        #S = Float64[]
                    else
                        # just shrink the TR
                        delta = gamma_dec*norm(s)
                    end
                else # we can't increase the precision anymore, so just reject
                    delta = gamma_dec*norm(s)
                end
            else
                if theta > eta_good*(-g'*s - 0.5*s'*H*s) && prec_lvl < num_prec && delta < min(1.0,norm(g))
                    temp_prec = prec_vec[prec_lvl + 1]
                    println("Probed a pair of function evaluations at ", temp_prec, " bits...")
                    ftemp, gtemp = handle(x,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1
                        prec_str = string(prec_str, "s")
                    elseif prec_lvl + 1 == 2
                        prec_str = string(prec_str, "d")
                    end
                    ftempplus,gtempplus = handle(x+s,temp_prec)
                    # hard-coded benchmarking:
                    if prec_lvl + 1 == 1
                        prec_str = string(prec_str, "s")
                    elseif prec_lvl + 1 == 2
                        prec_str = string(prec_str, "d")
                    end
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    if theta > eta_good*(-g'*s - 0.5*s'*H*s)
                        prec_lvl += 1
                        println("Permanently switching to precision level ", temp_prec, " bits:")
                        f = ftemp
                        g = gtemp
                        fplus = ftempplus
                        gplus = gtempplus
                        gprev = gtemp
                        delta = min(1.0,norm(g))
                        # testing:
                        #Y = []
                        #S = []
                    end
                else # the model quality isn't suspect, standard reject
                    delta = gamma_dec*delta
                end
            end
        else # the step is at least good
            if first_success == 0
                first_success = 1
            end
            xnew = x + s
            f = fplus
            g = gplus
            # is the step great?
            if norm(s) > 0.8*delta && rho > eta_great
                delta = min(max_delta,gamma_inc*delta)
            else
                delta = norm(s)
            end
        end

        if first_success == 1
            y0 = gplus-gprev
            #constant = (y0'*s)/(y0'*y0)
            constant = 1.0
            B0 = constant*B0
            H = B0
            first_success = 2
        end

        # update the Hessian
        if first_success > 0
            H,Y,S = update_hessian(H,B0,Y,S,gplus,gprev,s,memory,sr1_tol)
        end

        # get ready for next iteration
        x = xnew

        # TODO: write better output to command line indicating progress
        println("iter: ", iter, ", f: ",f, ", delta: ", delta, ", norm(g): ", norm(g))
        if isnan(f) || isnan(delta)
            break
        end
        # early termination if progress seems unlikely
        if norm(g) < epsilon
            if prec_lvl == num_prec
                return prechist, xhist
            #else
            #    prec_lvl += 1
            #    prec = prec_vec[prec_lvl]
            #    println("Function evaluations at precision level ", prec_vec[prec_lvl], " bits:")
            #    f,g = handle(x,prec)
            end
        end
        iter += 1
        # end of iteration benchmarking:
        push!(prechist,prec_str)
        xhist[iter,:] = x'
    end # end main for loop
    return prechist, xhist
end # end DynTR

function update_hessian(H::Matrix,B0::Matrix,Y::Array,S::Array,gplus::Array,gprev::Array,s::Array,memory::Integer,sr1_tol::Float64)
    y = gplus-gprev
    pred_grad = y - H*s
    dot_prod = pred_grad'*s
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s)
        if ~isempty(Y)
        # the dot product is large enough, update shouldn't cause instability
            if size(Y,2) >= memory
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,2:memory]
                S = S[:,2:memory]
            end
        end
        # add the newest (y,s) pair
        if isempty(Y)
            Y = y
            S = s
        else
            Y = hcat(Y,y)
            S = hcat(S,s)
        end

        Psi = Y-B0*S
        SY = S'*Y

        if typeof(SY)!=Float64
            D = Diagonal(SY)
            L = LowerTriangular(SY) - D
            U = UpperTriangular(SY) - D
            M = try
                inv(D+L+L' - S'*B0*S )
            catch singularerr
                n = length(y)
                inv(D+L+L' - S'*B0*S + sr1_tol*Matrix{Float64}(I,size(Y,2),size(Y,2)))
            end

        else
            M = 1.0/(SY - S'*B0*S)
        end
        H = real(B0 + Psi*M*Psi')
    end

    return H,Y,S
end #end update_hessian

end # end module
