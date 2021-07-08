import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

eps = 1.0e-6
max_problem_dim = 100
file_path = '/Users/clancy/repos/trophy/python/data/'
sin_path = file_path + 'single_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
dou_path = file_path + 'double_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
dyn_path = file_path + 'dynTR_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
lbf_path = file_path + 'lbfgs_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
tr_path = file_path + 'trncg_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'


sin = pd.read_csv(sin_path)
dou = pd.read_csv(dou_path)
dyn = pd.read_csv(dyn_path)
lbf = pd.read_csv(lbf_path)
tr = pd.read_csv(tr_path)

dfs = [sin, dou, dyn, lbf, tr]
for df in dfs:
    df['adjusted_evals'] = (df['fevals'] - df['single_evals']) + 0.5*df['single_evals']


title_list = ['Time (eps=' + str(eps)+')', 'Adjusted func. evals.', 'Total func. evals.', 'Iterations']

solver_list = ['Single TR', 'Double TR', 'Dynamic TR', 'L-BFGS', 'TR-NCG']

profile_for = ['time','adjusted_evals','fevals','nits']

#choose field to profile
#['problem', 'success', 'time', 'feval', 'gradnorm', 'nits', 'fevals','message', 'adjusted_evals']
#profile_field = 'gradnorm'
#profile_field = 'time'
#profile_field = 'nits'
#profile_field = 'nfevals'

# detemine successful runs and if they aren't set values to inf
temp = np.array([sin['success'], dou['success'],dyn['success'], lbf['success'], tr['success']]).T
SUCCESS = (temp == 'converged')

for ii, prof in enumerate(profile_for):
    M = np.array([sin[prof], dou[prof],dyn[prof], lbf[prof], tr[prof]]).T
    M = np.double(M)
    M[np.isnan(M)] = np.inf
    M[np.logical_not(SUCCESS)]= np.inf

    # construct matrix
    #M_min = np.min(M,1)


    for i, m in enumerate(M):
        temp = np.unique(m)
        if len(temp) > 1: m[m==0] = temp[1]
        if sum(m) == 0:
            M[i,:] = np.ones(m.shape)
        else:
            M[i,:] = m

    # construct matrix

    M_min = np.min(M,1)


    M = (M.T/M_min).T
    t = np.unique(M)
    t = t[t != np.inf]
    idx = np.logical_not(np.isnan(t))
    t = t[idx]

    n_probs = M.shape[0]
    C = np.zeros((t.shape[0], M.shape[1]))
    for (i, thresh) in enumerate(t):
        for j in range(M.shape[1]):
            temp = M[SUCCESS[:,j],j]
            C[i,j] = sum(vals <= thresh for vals in temp)/n_probs

    plt.subplot(2,2,ii+1)
    for (solver, c) in zip(solver_list, C.T):
        remove_last = 4
        plt.semilogx(t[:-remove_last],c[:-remove_last], label=solver)
    plt.ylim([0,1])
    plt.xlim([1,t[-remove_last-1]])

    plt.title(title_list[ii])

plt.legend(loc='lower right')
plt.show()