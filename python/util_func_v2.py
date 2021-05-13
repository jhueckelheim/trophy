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



def updateYS(Y, S, y, s, memory, sr1_tol = 1e-8):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - Hessian_times_vec(Y,S,s)
    dot_prod = np.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):
        if np.sum(Y) != 0:
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]
            #gamma = (Y[:,-1]@Y[:,-1]) / (S[:,-1]@Y[:,-1])
            #A = (Y-gamma*S).T@(Y-gamma*S)
            #print(LA.eig(A)[0])



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
        gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
        L = np.zeros((Y.shape[1],Y.shape[1]))
        d = np.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            d[ii] = S[:,ii].T@Y[:,ii]
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].T@Y[:,jj]

        D = np.diag(d)
        M = D + L + L.T - gamma*(S.T@S)
        Minv = LA.inv(M)
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        Y = np.zeros((nv,1))
        S = np.zeros((nv,1))

    ################  Bk is approximation of Hessian...not it's inverse.
    tmp1 = (Y - gamma*S).T@v
    tmp2 = Minv@tmp1
    B_v = (Y - gamma*S)@tmp2 + gamma*v
    return B_v



def CG_Steinhaug_matFree(eps, g, delta, S, Y):
    nv = len(g)
    zOld = np.zeros((nv,1))
    rOld = g
    dOld = -g
    keep_going = True

    if norm(rOld) < eps:
        p = zOld
        return p
        #keep_going = False

    # use compact limited form to generate
    if np.sum(Y) != 0 or np.sum(S) != 0:
        gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
        L = np.zeros((Y.shape[1],Y.shape[1]))
        d = np.zeros(L.shape[0])
        for ii in range(Y.shape[1]):
            d[ii] = S[:,ii].T@Y[:,ii]
            for jj in range(0,ii):
                L[ii,jj] = S[:,ii].T@Y[:,jj]

        D = np.diag(d)
        M = L + L.T + D - gamma*(S.T@S)
        Minv = LA.inv(M)
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        Y = np.zeros((nv,1))
        S = np.zeros((nv,1))


    j = 0
    while keep_going:
        temp1 = (Y-gamma*S).T@dOld
        temp2 = Minv@temp1
        B_dOld = (Y-gamma*S)@temp2 + gamma*dOld

        dBd = dOld.T@B_dOld
        # does direction have negative curvature?
        if dBd <= 1e-12:
            # find tau that gives minimizer
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            #keep_going = False
            return p

        alphaj = norm(rOld)**2 / dBd
        zNew = zOld + alphaj*dOld
        if norm(zNew) >= delta:
            tau = rootFinder(norm(dOld)**2, 2*zOld.T@dOld, (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            return p

        rNew = rOld + alphaj*B_dOld
        if norm(rNew) <= eps or j > nv:
            p = zNew
            return p
            #keep_going = False

        betaNew = norm(rNew)**2/norm(rOld)**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        j += 1

        #print('CG it ', j)
        if j > 10000:
            print('Going too long')





    #return p



def BFGS_hessian_times_vec(Y,S,v):
    # Letting Hessian remain V to agree with convention elsewhere.
    nv = len(v)
    if np.sum(Y) != 0:
        gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
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
            print('Matrix is singular')
            Minv = LA.inv(M + 1e-6*np.eye(M.shape[0]))
            time.sleep(10)
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        A = np.zeros((nv,1))

    ################  Bk is approximation of Hessian...not it's inverse.
    tmp1 = A.T@v
    tmp2 = Minv@tmp1
    B_v = gamma*v - A@tmp2
    return B_v



def CG_Steinhaug_BFGS_matFree(eps, g, delta, S, Y):
    nv = len(g)
    zOld = np.zeros((nv,1))
    rOld = g
    dOld = -g
    keep_going = True

    if norm(rOld) < eps:
        p = zOld
        return p

    # use compact limited form to generate
    if np.sum(Y) != 0 or np.sum(S) != 0:
        gamma = (Y[:,-1].T@Y[:,-1]) / (S[:,-1].T@Y[:,-1])
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
            Minv = LA.inv(M + 1e-6*np.eye(M.shape[0]))
    else:
        gamma = 1
        Minv = np.zeros((1,1))
        A = np.zeros((nv,1))


    j = 0
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
                print('CG should have converged by now')
                return p, "Too many iterations"
            else:
                return p, "Success in TR"

        betaNew = norm(rNew)**2/norm(rOld)**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        j += 1










def BFGS_updateYS(Y, S, y, s, memory, sr1_tol = 1e-8):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x
    :param memory: memory allocation
    :param sr1_tol: helps determine when to update Y and S
    :return: updated Y and S
    """
    pred_grad = y - BFGS_hessian_times_vec(Y,S,s)
    dot_prod = np.dot(pred_grad,s)
    if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):
    #if np.abs(np.dot(s,y)) > 1e-16: # and abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):
        if np.sum(Y) != 0:
            # the dot product is large enough, update shouldn't cause instability
            if Y.shape[1] >= memory:
                # all memory used up, delete the oldest (y,s) pair
                Y = Y[:,1:memory]       # since indexed from zero, start at 1 not 2 and go to memory
                S = S[:,1:memory]
            #gamma = (Y[:,-1]@Y[:,-1]) / (S[:,-1]@Y[:,-1])
            #A = (Y-gamma*S).T@(Y-gamma*S)
            #print(LA.eig(A)[0])



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
