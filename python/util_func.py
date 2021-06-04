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
import math

# ==========================================================================
def CG_Steinhaug_matFree_OLD(epsTR, g , deltak, S,Y,nv):
    """
    :param g: gradient
    :param deltak: TR radius
    :param S: difference of x's
    :param Y: differences of gradients
    :param nv: dimension of search space
    :return: search direction
    The following function is used for sloving the trust region subproblem
    by utilizing "CG_Steinhaug" algorithm discussed in
    Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.
    (actually in chapter 7 under trust region newton-CG method pp. 170);
    moreover, for Hessian-free implementation, we used the compact form of Hessian
    approximation discussed in Byrd, Richard H., Jorge Nocedal, and Robert B. Schnabel.
    "Representations of quasi-Newton matrices and their use in limited memory methods."
    Mathematical Programming 63.1-3 (1994): 129-156

    """
    zOld = np.zeros((nv,1))
    rOld = g
    dOld = -g
    trsLoop = 1e-12
    if LA.norm(rOld) < epsTR:
        return zOld
    flag = True
    pk= np.zeros((nv,1))

    # for Hessfree
    L = np.zeros((Y.shape[1],Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0,ii):
            L[ii,jj] = S[:,ii].dot(Y[:,jj])


    tmp = np.sum((S * Y),axis=0)  # gives diagonal of S^T@Y

    D = np.diag(tmp)
    M = (D + L + L.T)
    #try:
    Minv = np.linalg.inv(M)
    #except:
        #Minv = np.linalg.inv(M + (1e-6)*np.eye(M.shape[0]))
    #    print('Warning!!! Matrix indefinite, added diagonal matrix to impose SPD')

    while flag:

        ################  Bk is approximation of Hessian...not it's inverse.
        tmp1 = np.matmul(Y.T,dOld)
        tmp2 = np.matmul(Minv,tmp1)
        Bk_d = np.matmul(Y,tmp2)

        ################

        if dOld.T.dot(Bk_d) < trsLoop:
            tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
            pk = zOld + tau*dOld
            flag = False
            break
        alphaj = rOld.T.dot(rOld) / (dOld.T.dot(Bk_d))
        zNew = zOld +alphaj*dOld

        if LA.norm(zNew) >= deltak:
            tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
            pk = zOld + tau*dOld
            flag = False
            break
        rNew = rOld + alphaj*Bk_d

        if LA.norm(rNew) < epsTR:
            pk = zNew
            flag = False
            break
        betajplus1 = rNew.T.dot(rNew) /(rOld.T.dot(rOld))
        dNew = -rNew + betajplus1*dOld

        zOld = zNew
        dOld = dNew
        rOld = rNew
    return pk

# ==========================================================================
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

def L_BFGS_two_loop_recursion_OLD(g_k,S,Y,k,mmr,gamma_k,nv):
    """
    The following function returns the serach direction based
    on LBFGS two loop recursion discussed in
    Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.
    """
    #   idx = min(k,mmr)
    idx = min(S.shape[1],mmr)
    rho = np.zeros((idx,1))

    theta = np.zeros((idx,1))
    q = g_k
    for i in xrange(idx):
        rho[idx-i-1] = 1/ S[:,idx-i-1].reshape(nv,1).T.dot(Y[:,idx-i-1].reshape(nv,1))
        theta[idx-i-1] =(rho[idx-i-1])*(S[:,idx-i-1].reshape(nv,1).T.dot(q))
        q = q - theta[idx-i-1]*Y[:,idx-i-1].reshape(nv,1)

    r = gamma_k*q
    for j in xrange(idx):
        beta = (rho[j])*(Y[:,j].reshape(nv,1).T.dot(r))
        r = r + S[:,j].reshape(nv,1)*(theta[j] - beta)

    return r

def update_hessian_OLD(H, B0, Y, S, gplus, gprev, s, memory, sr1_tol):
    y = gplus-gprev
    pred_grad = y - H@s
    dot_prod = pred_grad@s
    if abs(dot_prod) > sr1_tol*LA.norm(pred_grad)*LA.norm(s):
        if Y.shape[0] > 0:
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]


        # add the newest (y,s) pair
        if Y.shape[0] == 0: #not bool(Y):
            Y = y
            S = s
        else:
            Y = np.hstack((Y,y.reshape((y.shape[0],1))))
            S = np.hstack((S,s.reshape((s.shape[0],1))))
            #Y = np.hstack((Y,y))
            #S = np.hstack((S,s))


        Psi = Y-B0@S
        SY = S.T@Y

        #if typeof(SY)!=Float64
        if not isinstance(SY, np.float64):   # check to see if matrix
        #if SY.shape[0] != 1:   # check to see if matrix
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
            Psi = Psi.reshape((len(Psi),1))
            M = M.reshape((1,1))

        H = np.real(B0 + (Psi@M)@Psi.T)

    return H, Y, S

def Hessian_times_vec_OLD(Y,S,v):
    # Letting Hessian remain V to agree with convention elsewhere.
    L = np.zeros((Y.shape[1],Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0,ii):
            L[ii,jj] = S[:,ii].dot(Y[:,jj])

    tmp = np.sum((S * Y),axis=0)  # gives diagonal of S^T@Y
    D = np.diag(tmp)
    M = (D + L + L.T)
    #try:
    Minv = np.linalg.inv(M)
    #except:
    #Minv = np.linalg.inv(M + (1e-6)*np.eye(M.shape[0]))
    #print('Singular M for Hessian times a vec')

    tmp1 = np.matmul(Y.T,v)
    tmp2 = np.matmul(Minv,tmp1)
    B_v = np.matmul(Y,tmp2)

    return B_v









def updateYS(Y, S, y, s, memory, sr1_tol = 1e-3):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param gnew: latest gradient calculation
    :param gold: previous gradient calculation
    :param s: most recent step direction
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - Hessian_times_vec(Y,S,s)
    dot_prod = np.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*LA.norm(pred_grad)*LA.norm(s):
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
        print('Not updating Y and S')

    return Y, S



def Hessian_times_vec(Y,S,v):
    # Letting Hessian remain V to agree with convention elsewhere.
    nv = len(v)

    if np.sum(Y) != 0:
        gamma = (Y[:,-1]@Y[:,-1]) / (S[:,-1]@Y[:,-1])
        L = np.zeros((Y.shape[1],Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].dot(Y[:,jj])

        tmp = np.sum((S * Y),axis=0)  # gives diagonal of S^T@Y
        D = np.diag(tmp)
        M = D + L + L.T - gamma*(S.T@S)
        Minv = np.linalg.inv(M)
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        Y = np.zeros((len(v),1))
        S = np.zeros((len(v),1))

    Y_minus_BS = Y - gamma*S
    ################  Bk is approximation of Hessian...not it's inverse.
    tmp1 = np.matmul( Y_minus_BS.T,v)
    tmp2 = np.matmul(Minv,tmp1)
    B_v = np.matmul(Y_minus_BS,tmp2) + gamma*v

    return B_v



def CG_Steinhaug_matFree(epsTR, g , deltak, S, Y):
    """
    :param epsTR: accuracy for which to solve TR subproblem
    :param g: gradient
    :param deltak: TR radius
    :param S: difference of x's
    :param Y: differences of gradients
    :param nv: dimension of search space
    :return: search direction
    The following function is used for sloving the trust region subproblem
    by utilizing "CG_Steinhaug" algorithm discussed in
    Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.
    (actually in chapter 7 under trust region newton-CG method pp. 170);
    Richie Clancy edits

    """
    nv = len(g)
    zOld = np.zeros((nv,1))
    rOld = g
    dOld = -g
    trsLoop = 1e-12
    if LA.norm(rOld) < epsTR:
        # if gradient is already small, we're done.
        return zOld
    flag = True
    pk= np.zeros((nv,1))


    # Use B0 = gamma_k*I where gamma_k = (y_{k-1}^T y_{k-1}) / (s_{k-1}^T y_{k-1})
    if np.sum(Y) > 0:
        # for gamma by taking dot product from last columns of Y and S
        gamma = (Y[:,-1]@Y[:,-1]) / (S[:,-1]@Y[:,-1])
        # for Hessfree
        L = np.zeros((Y.shape[1],Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].dot(Y[:,jj])


        tmp = np.sum((S * Y),axis=0)  # gives diagonal of S^T@Y
        D = np.diag(tmp)
        M = D + L + L.T - gamma*(S.T@S)
        #try:
        Minv = np.linalg.inv(M)
        #except:
        #Minv = np.linalg.inv(M + (1e-6)*np.eye(M.shape[0]))
        #    print('Warning!!! Matrix indefinite, added diagonal matrix to impose SPD')
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        Y = np.zeros((nv,1))
        S = np.zeros((nv,1))

    Y_minus_B0S = Y-gamma*S
    while flag:
        ################  Bk is approximation of Hessian...not it's inverse.
        tmp1 = np.matmul( Y_minus_B0S.T,dOld)
        tmp2 = np.matmul(Minv,tmp1)
        Bk_d = np.matmul(Y_minus_B0S,tmp2) + gamma*dOld

        ################

        if dOld.T.dot(Bk_d) < trsLoop:
            tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
            pk = zOld + tau*dOld
            flag = False
            break
        alphaj = rOld.T.dot(rOld) / (dOld.T.dot(Bk_d))
        zNew = zOld +alphaj*dOld

        if LA.norm(zNew) >= deltak:
            tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
            pk = zOld + tau*dOld
            flag = False
            break
        rNew = rOld + alphaj*Bk_d

        if LA.norm(rNew) < epsTR:
            pk = zNew
            flag = False
            break
        betajplus1 = rNew.T.dot(rNew) /(rOld.T.dot(rOld))
        dNew = -rNew + betajplus1*dOld

        zOld = zNew
        dOld = dNew
        rOld = rNew
    return pk





