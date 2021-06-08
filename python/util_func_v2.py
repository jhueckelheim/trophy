#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019 Albert Berahas, Majid Jahani, Martin Takáč
#
# All Rights Reserved.
#
# Authors: Albert Berahas, Majid Jahani, Martin Takáč
#
# Please cite:
#
#   A. S. Berahas, M. Jahani, and M. Takáč, "Quasi-Newton Methods for
#   Deep Learning: Forget the Past, Just Sample." (2019). Lehigh University.
#   http://arxiv.org/abs/1901.09997
# ==========================================================================

import numpy as np
import random
from numpy import linalg as LA
from numpy.linalg import norm
import math
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm
from jax.ops import index, index_add, index_update

dbug = False
# ==========================================================================

def pycutest_wrapper(x,prec, single_func, double_func):
    if prec == 1:
        f, g = single_func.obj(x, gradient=True)
    elif prec == 2:
        f, g = double_func.obj(x, gradient=True)
    else:
        print('Only recognize single or double for pyCUTEst')
    return f, g


def rootFinder(a,b,c):
    """return the root of (a * x^2) + b*x + c =0"""
    r = b**2 - 4*a*c

    if r > 0:
        num_roots = 2
        x1 = ((-b) + np.sqrt(r))/(2*a+0.0)
        x2 = ((-b) - np.sqrt(r))/(2*a+0.0)
        x = max(x1,x2)
        if x>=0:
            return x
        else:
            print("no positive root!")
    elif r == 0:
        num_roots = 1
        x = (-b) / (2*a+0.0)
        if x>=0:
            return x
        else:
            print("no positive root!")
    else:
        print("No roots")


def updateYS(Y, S, y, s, memory, gamma, sr1_tol = 1e-4, verbose=False):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - Hessian_times_vec(Y,S,gamma, s)
    dot_prod = np.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):
        if not isinstance(Y, list):
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]


        # add newest (y,s) pair
        if isinstance(Y, list):
            Y = y.reshape((y.shape[0],1))
            S = s.reshape((s.shape[0],1))
        else:
            Y = np.hstack((Y,y.reshape((y.shape[0],1))))
            S = np.hstack((S,s.reshape((s.shape[0],1))))



        keep_going = True
        while keep_going:
            if S.shape[1] > 0:
                Psi = Y - gamma*S
                Minv = np.matmul(S.T, Y) - gamma*np.matmul(S.T, S)
                tmp = np.min(LA.eig(np.matmul(Psi.T, Psi))[0])
                if tmp > 0 and LA.det(Minv) != 0:
                    keep_going = False
                else:
                    S = S[:, 1::]
                    Y = Y[:, 1::]
            else:
                keep_going = False

    else:
        if verbose: print('Not updating Y and S')

    return Y, S

def Hessian_times_vec(Y, S, gamma, v):
    # Letting Hessian remain V to agree with convention elsewhere.
    nv = len(v)
    if not isinstance(Y, list):
        ''' After timing, this is very slow especially for large matrices
        L = np.zeros((Y.shape[1],Y.shape[1]))
        R = np.zeros((S.shape[1],S.shape[1]))
        d = np.zeros(L.shape[0])
        r = np.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            d[ii] = np.dot(S[:, ii], Y[:, ii])
            r[ii] = np.dot(S[:, ii], S[:, ii])
            for jj in range(0, ii):
                L[ii, jj] = np.dot(S[:, ii], Y[:, jj])
                R[ii, jj] = np.dot(S[:, ii], S[:, jj])
         
        M = np.diag(d) + L + L.T - gamma*(np.diag(r) + R + R.T)  # np.dot(S.T, S)
        '''
        # suprisingly much faster for large matrices
        temp1 = np.matmul(S.T, Y)
        M = np.tril(temp1) + np.triu(temp1.T, 1) - gamma*np.matmul(S.T, S)
        try:
            Minv = LA.inv(M)
        except:
            Minv = LA.pinv(M)

    else:
        #gamma = 1
        Minv = np.zeros((1, 1))
        Y = np.zeros((nv, 1))
        S = np.zeros((nv, 1))

    ################  Bk is approximation of Hessian...not it's inverse.
    G = (Y - gamma*S)
    tmp1 = np.matmul(G.T, v)
    tmp2 = np.dot(Minv, tmp1)
    B_v = np.matmul(G, tmp2) + gamma*v
    return B_v


def CG_Steinhaug_matFree(eps, g, delta, S, Y, gamma, verbose=False, max_it=None):

    nv = len(g)
    if max_it is None:
        max_it = 3*nv

    zOld = np.zeros((nv, 1))
    try:
        g.shape[1]
    except:
        g = g.reshape((nv, 1))
    rOld = g
    dOld = -g
    keep_going = True

    if norm(rOld) < eps:
        p = zOld
        return p, "small residual"

    # use compact limited form to generate
    if not isinstance(Y, list):
        temp1 = np.matmul(S.T, Y)
        M = np.tril(temp1) + np.triu(temp1.T, 1) - gamma*np.matmul(S.T, S)
        try:
            Minv = LA.inv(M)
        except:
            Minv = LA.pinv(M)

    else:
        Minv = np.zeros((1, 1))
        Y = np.zeros((nv, 1))
        S = np.zeros((nv, 1))


    j = 0
    G = Y-gamma*S
    G_T = np.array(G.T, order='C')
    while keep_going:
        temp1 = np.matmul(G_T, dOld)
        temp2 = np.matmul(Minv, temp1)
        B_dOld = np.matmul(G, temp2) + gamma*dOld

        dBd = np.matmul(dOld.T, B_dOld)
        # does direction have negative curvature?
        if dBd <= 0:
            # find tau that gives minimizer
            tau = rootFinder(norm(dOld)**2, 2*np.dot(zOld.T, dOld), (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld

            # this is where problems occur. Check to see if there is a descent along this particular direction
            # note that g'*p + 0.5*p'*B*p should be negative.
            if dbug:
                temp1 = np.matmul(G_T, p)
                temp2 = np.matmul(Minv, temp1)
                B_p = (G)@temp2 + gamma*p
                B_p = np.matmul(G, temp2) + gamma*p
                model_change = np.dot(g.T, p) + 0.5*np.dot(p.T, B_p)
                if model_change > 0:
                    # form approximate Hessian to probe eigenvalues, etc
                    if True:
                        B = np.matmul(G,np.matmul(Minv,G_T)) + gamma*np.eye(nv)
                        model_func = lambda z: g.T@z + 0.5*z.T@B@z
                        print("Model increase of", model_change[0][0],  ", recent gradient is", norm(rOld), "and cond(B)=",LA.cond(B))
                        ev, E = LA.eig(B)
                        idx = np.argsort(ev)
                        ev = ev[idx]
                        E = E[:,idx]


                    tauk = tau
                    tau_tol = 1e-16
                    while tauk > tau_tol:
                        pk = zOld + tauk*dOld

                        temp1 = G.T@pk
                        temp2 = Minv@temp1
                        B_pk = G@temp2 + gamma*pk
                        model_change = model_func(pk)
                        model_change = g.T@pk + 0.5*pk.T@B_pk

                        tauk = tauk*.1
                        print('p.T*B*p=',pk.T@B_pk,'Model change is', model_change[0][0], 'and line search tau=', tauk)
                        #print('M(x)=',g.T@zOld + 0.5*zOld.T@B@zOld)
                        print('tau*g.T@d + 0.5*tau^2*d.T@B@d=',tauk*g.T@dOld + 0.5*tauk**2*dOld.T@B@dOld)
                        print(tauk*(zOld.T@B-g.T)@dOld + 0.5*(tauk**2)*dBd)
                        #print('TxBd=',tauk*zOld.T@B@dOld)
                    if tau < tau_tol:
                        print('Tau = arg m(p = z+tau*d) is less than', tau_tol, 'in CG sub problem')

            if dBd == 0:
                print("The matrix is indefinite") if verbose else None
            return p, "neg. curve",

        alphaj = norm(rOld)**2 / dBd
        zNew = zOld + alphaj*dOld
        if norm(zNew) >= delta:
            tau = rootFinder(norm(dOld)**2, 2*np.dot(zOld.T, dOld), (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            return p, "exceed TR"

        rNew = rOld + alphaj*B_dOld
        if norm(rNew) <= eps or j > max_it:  # or norm(zNew - zOld) <= eps
            p = zNew
            if norm(rNew) > eps:
                print('CG should have converged by now') if verbose else None
                return p, "Too many CG iterations"
            else:
                return p, "Success in TR"

        betaNew = norm(rNew)**2/norm(rOld)**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        j += 1


class structtype():
    # pulled from https://stackoverflow.com/questions/11637045/complex-matlab-like-data-structure-in-python-numpy-scipy
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val
    def ListVariables(self):
        names = dir(self)
        for name in names:
            # Print the item if it doesn't start with '__'
            if  '__' not in name and 'Set' not in name and 'SetAttr' not in name and 'ListVariables' not in name:
                myvalue = self.__dict__[name]
                print(name, ':', myvalue)



























def Hessian_times_vec_JAX(Y, S, gamma, v):
    # Letting Hessian remain V to agree with convention elsewhere.
    nv = len(v)

    if not isinstance(Y, list):
        L = jnp.zeros((Y.shape[1],Y.shape[1]))
        sig = jnp.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            sig = index_update(sig, index[ii], jnp.dot(S[:, ii], Y[:, ii]))
            for jj in range(0,ii):
                L = index_update(L, index[ii, jj], jnp.dot(S[:,ii], Y[:,jj]))

        D = jnp.diag(sig)
        M = D + L + L.T - gamma*jnp.matmul(S.T, S)
        try:
            Minv = jnp.linalg.inv(M)
        except:
            Minv = jnp.linalg.pinv(M)
    else:
        Minv = jnp.zeros((1,1))
        Y = jnp.zeros((nv,1))
        S = jnp.zeros((nv,1))

    ################  Bk is approximation of Hessian...not it's inverse.
    tmp1 = jnp.matmul( (Y - gamma*S).T, v)
    tmp2 = jnp.matmul(Minv, tmp1)
    B_v = jnp.matmul(Y - gamma*S, tmp2) + gamma*v
    return B_v


def CG_Steinhaug_matFree_JAX(eps, g, delta, S, Y, gamma, verbose=False):
    nv = len(g)
    zOld = jnp.zeros((nv, 1))
    try:
        g.shape[1]
    except:
        g = g.reshape((nv, 1))
    g = jnp.array(g)
    rOld = g
    dOld = -g
    keep_going = True

    if jnp.linalg.norm(rOld) < eps:
        p = zOld
        return p, "small residual"


    if not isinstance(Y, list):
        L = jnp.zeros((Y.shape[1], Y.shape[1]))
        sig = jnp.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            #sig[ii] = jax.ops.index_update( sig, index[ii], jnp.dot(S[:, ii], Y[:, ii]))
            sig = index_update(sig, index[ii], jnp.dot(S[:, ii], Y[:, ii]))
            for jj in range(0, ii):
                #L[ii, jj] = jnp.dot(S[:, ii], Y[:, jj])
                L = index_update(L, index[ii,jj], jnp.dot(S[:, ii], Y[:, jj]))

        D = jnp.diag(sig)
        M = L + L.T + D - gamma*jnp.matmul(S.T, S)
        try:
            Minv = jnp.linalg.inv(M)
        except:
            Minv = jnp.linalg.pinv(M)
    else:
        Minv = jnp.zeros((1, 1))
        Y = jnp.zeros((nv, 1))
        S = jnp.zeros((nv, 1))


    j = 0
    while keep_going:
        temp1 = jnp.matmul( (Y-gamma*S).T, dOld)
        temp2 = jnp.matmul(Minv, temp1)
        B_dOld = jnp.matmul(Y-gamma*S, temp2) + gamma*dOld

        dBd = jnp.matmul(dOld.T,B_dOld)
        # does direction have negative curvature?
        if dBd <= 0:
            # find tau that gives minimizer
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld

            if dBd == 0:
                if verbose: print("The matrix is indefinite")

            return p, "neg. curve",

        alphaj = norm(rOld)**2 / dBd
        zNew = zOld + alphaj*dOld
        if norm(zNew) >= delta:
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            return p, "exceed TR"

        rNew = rOld + alphaj*B_dOld
        if norm(rNew) <= eps or j > 3*nv:  # or norm(zNew - zOld) <= eps
            p = zNew
            if norm(rNew) > eps:
                if verbose: print('CG should have converged by now')
                return p, "Too many CG iterations"
            else:
                return p, "Success in TR"

        betaNew = norm(rNew)**2/norm(rOld)**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        j += 1


def updateYS_JAX(Y, S, y, s, memory, gamma, sr1_tol = 1e-4, verbose=False):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - Hessian_times_vec_JAX(Y,S,gamma, s)
    dot_prod = jnp.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*jnorm(s):
        if not isinstance(Y, list):
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:, 1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:, 1:memory]


        if isinstance(Y, list):
            Y = y.reshape((y.shape[0],1))
            S = s.reshape((s.shape[0],1))
        else:
            Y = np.hstack((Y,y.reshape((y.shape[0],1))))
            S = np.hstack((S,s.reshape((s.shape[0],1))))

        keep_going = True
        while keep_going:
            if S.shape[1] > 0:
                Psi = Y - gamma*S
                Minv = jnp.matmul(S.T, Y) - gamma*jnp.matmul(S.T, S)
                tmp = min(jnp.linalg.eig( jnp.matmul(Psi.T, Psi))[0])
                if tmp > 0 and jnp.linalg.det(Minv) != 0:
                    keep_going = False
                else:
                    S = S[:, 1::]
                    Y = Y[:, 1::]
            else:
                keep_going = False
    else:
        if verbose: print('Not updating Y and S')

    return Y, S


def BFGS_hessian_times_vec(Y,S,gamma,v,verbose=False):
    # Letting Hessian remain V to agree with convention elsewhere.
    nv = len(v)
    if np.sum(Y) != 0:
        #gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
        L = np.zeros((Y.shape[1],Y.shape[1]))
        temp = np.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            temp[ii] = S[:,ii].T@Y[:,ii]
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].T@Y[:,jj]

        A = np.hstack((gamma*S, Y))
        topmat = np.hstack((gamma*S.T@S, L))
        botmat = np.hstack((L.T, -np.diag(temp)))

        M = np.vstack( (topmat,botmat) )
        try:
            Minv = LA.inv(M)
        except:
            if verbose: print('Matrix is singular')
            Minv = LA.inv(M + 1e-6*np.eye(M.shape[0]))
            time.sleep(10)
    else:
        #gamma = 1
        Minv = np.zeros((1,1))
        A = np.zeros((nv,1))

    ################  Bk is approximation of Hessian...not it's inverse.
    tmp1 = A.T@v
    tmp2 = Minv@tmp1
    B_v = gamma*v - A@tmp2
    return B_v


def BFGS_CG_Steinhaug_matFree(eps, g, delta, S, Y, gamma, verbose=False):
    nv = len(g)
    zOld = np.zeros((nv,1))
    rOld = g
    dOld = -g
    keep_going = True

    if norm(rOld) < eps:
        p = zOld
        return p, "small residual"

    # use compact limited form to generate
    if np.sum(Y) != 0 or np.sum(S) != 0:
        #gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
        L = np.zeros((Y.shape[1],Y.shape[1]))
        temp = np.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            temp[ii] = S[:,ii].T@Y[:,ii]
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].T@Y[:,jj]

        A = np.hstack((gamma*S, Y))
        topmat = np.hstack((gamma*S.T@S, L))
        botmat = np.hstack((L.T, -np.diag(temp)))

        M = np.vstack( (topmat,botmat) )

        if abs(LA.det(M)) > 1.e-16:
            Minv = LA.inv(M)
        else:
            Minv = LA.inv(M + 1e-6*np.eye(M.shape[0]))
    else:
        Minv = np.zeros((1,1))
        A = np.zeros((nv,1))

    j = 0

    if dbug:
        Bk = gamma*np.eye(nv) - A@Minv@A.T
        #Bk = (Bk + Bk.T)/2.
        lam = np.min(LA.eig(Bk)[0])
        if lam > 0:
            #print('Matrix is positive definite with lambda_min=', lam)
            pass
        elif lam == 0:
            pass
            #print('Matrix is singular')
        else:
            print('Matrix is indefinite with e.vals', LA.eig(Bk)[0])
            print(gamma)

    while keep_going:
        tmp1 = A.T@dOld
        try:
            tmp2 = Minv@tmp1
        except:
            print('Check problem out')

        B_dOld = gamma*dOld - A@tmp2

        dBd = dOld.T@B_dOld
        # does direction have negative curvature?
        if dBd <= 0:
            # find tau that gives minimizer
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            if dBd == 0:
                if verbose: print("The matrix is indefinite")
            return p, "neg. curve",

        alphaj = norm(rOld)**2 / dBd
        zNew = zOld + alphaj*dOld
        if norm(zNew) >= delta:
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            return p, "exceed TR"

        rNew = rOld + alphaj*B_dOld
        if norm(rNew) <= eps or j > 5*nv:
            p = zNew
            if norm(rNew) > eps:
                if verbose: print('CG should have converged by now')
                return p, "Too many iterations"
            else:
                return p, "Success in TR"

        betaNew = norm(rNew)**2/norm(rOld)**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        j += 1


def BFGS_updateYS(Y, S, y, s, memory, gamma, sr1_tol = 1e-4, verbose=False):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - BFGS_hessian_times_vec(Y,S,gamma,s)
    dot_prod = np.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s) and np.abs(np.dot(s,y)) > 1e-16:
        if np.sum(Y) != 0:
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]

        # add the newest (y,s) pair
        if np.sum(Y) == 0:
            Y = y.reshape((y.shape[0],1))
            S = s.reshape((s.shape[0],1))
        else:
            Y = np.hstack((Y,y.reshape((y.shape[0],1))))
            S = np.hstack((S,s.reshape((s.shape[0],1))))
    else:
        if verbose:
            print('Not updating Y and S')

    return Y, S

