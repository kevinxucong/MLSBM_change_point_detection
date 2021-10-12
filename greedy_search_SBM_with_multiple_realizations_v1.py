from VEM_SBM_with_multiple_realizations_v1 import *

def backward_elimination(init_breaks, Q_ls, A, tmax = 10, smax = 5, fpeps = 1e-04, emeps = 1e-10, TAUMIN = 1e-10, MSTEPPARAMMIN = 1e-6):
    """
    init_breaks: list, 1 <= init_breaks[i] < init_breaks[j] < T for all i < j
    """
    T, n, _ = A.shape
    
    # (t1, t2): MDL / community_assignment / pi / theta
    dict_mdl = {}
    dict_assignment = {}
    dict_pi = {}
    
    # number of time intervals: MDL / change points
    overall_dict_mdl = {}
    overall_dict_change_points = {}
    
    M = len(init_breaks) + 1
    cur_ls = [c for c in zip([0]+init_breaks, init_breaks+[T])]
    
    for it in range(M):
        if it == 0:
            working_ls = cur_ls[:]
        else:
            working_ls = [(c[0][0], c[1][1]) for c in zip(cur_ls[:-1], cur_ls[1:])]
        
        for l in range(len(working_ls)):
            if (working_ls[l][0], working_ls[l][1]) in dict_mdl:
                continue

            A1 = A[working_ls[l][0]: working_ls[l][1]]
            T1 = working_ls[l][1] - working_ls[l][0]

            res_mdl = float('inf')
            res_assignment = [0]*n
            res_pi = []
            for Q in Q_ls:
                init_ls = []
                init_ls.append(kmeans2(np.sum(A1, axis=0), Q, minit='points')[1])
                init_ls.append(kmeans2(np.max(A1, axis=0), Q, minit='points')[1])
                for t in np.random.choice(T1, np.minimum(T1, 1)):
                    init_ls.append(kmeans2(A1[t], Q, minit='points')[1])

                for init_labels in init_ls:
                    res = VEM(init_labels, A1, n, Q, T1, tmax, smax, fpeps, emeps, TAUMIN, MSTEPPARAMMIN)
                    Z_hat, pi_hat = get_MLE(res[0], res[2], A1, n, Q, T1)
                    mdl = MDL(Z_hat, pi_hat, A1, n, Q, T1)
                    if mdl < res_mdl:
                        res_mdl = float(mdl)
                        res_assignment = np.argmax(Z_hat,axis=1).tolist()
                        res_pi = pi_hat.tolist()
                        
            dict_mdl[(working_ls[l][0], working_ls[l][1])] = res_mdl
            dict_assignment[(working_ls[l][0], working_ls[l][1])] = res_assignment
            dict_pi[(working_ls[l][0], working_ls[l][1])] = res_pi
            
        if it == 0:
            overall_dict_mdl[M-it] = sum(dict_mdl[interval] for interval in working_ls) + np.log(M-it) + sum(np.log(t2-t1) for t1,t2 in working_ls)
            overall_dict_change_points[M-it] = working_ls[:]
            
        else:
            min_mdl = float('inf')
            best_ls = [0]*(M-it)
            for j in range(len(working_ls)):
                temp_ls = cur_ls[:j] + [working_ls[j]] + cur_ls[(j+2):]
                temp_mdl = sum(dict_mdl[interval] for interval in temp_ls) + np.log(M-it) + sum(np.log(t2-t1) for t1,t2 in temp_ls)
                if temp_mdl < min_mdl:
                    min_mdl = float(temp_mdl)
                    best_ls = temp_ls[:]
            overall_dict_mdl[M-it] = min_mdl
            overall_dict_change_points[M-it] = best_ls[:]
            
            cur_ls = best_ls[:]
            
    fitted_mdl = float('inf')
    fitted_change_points = []
    for m in range(1, M+1):
        if overall_dict_mdl[m] < fitted_mdl:
            fitted_mdl = overall_dict_mdl[m]
            fitted_change_points = overall_dict_change_points[m]
            
    fitted_assignment = [dict_assignment[interval] for interval in fitted_change_points]
    fitted_pi = [dict_pi[interval] for interval in fitted_change_points]
    temp = {"fitted_change_points": fitted_change_points,
            "fitted_mdl": fitted_mdl,
            "fitted_pi": fitted_pi,
            "fitted_assignment": fitted_assignment
            }
    return temp, dict_mdl, dict_assignment, dict_pi, overall_dict_mdl, overall_dict_change_points
