import numpy as np
import scipy.stats
import scipy.special

def spearman_permutation(table, cols, num_iters = [20,400,16000,100000,2000000], tocompare=['UV_Slope','dust2(50)','delta(50)','E_b(50)','log10age(50)','log10mass(50)']):
    #num_iters chose to get 2, 3, 4, 4.5, 5 sigma
    for i, colname1 in enumerate(cols):
        if i + 1 < len(cols):
            for j, colname2 in enumerate(cols[i+1:]):
                if (colname1 not in tocompare) and (colname2 not in tocompare):
                    continue
                totalmask = ~table[colname1].data.mask & ~table[colname2].data.mask
                col1 = table[colname1][totalmask]
                col2 = table[colname2][totalmask]
                orig_spearman = scipy.stats.spearmanr(col1,col2)[0]
                assert len(col1) == len(col2)

                print(colname1, colname2)
                for k in num_iters:
                    print(k)
                    spearman = spearman_helper(col1,col2,k)
                    p = len(spearman[np.abs(spearman) > np.abs(orig_spearman)]) / len(spearman)
                    if p > 4/k:
                        break
                        
                if p == 0:
                    p = 1.0/num_iters[-1]
                    signif = scipy.special.ndtri(1 - p/2)
                    p = '<1/'+str(num_iters[-1])
                else:
                    signif = scipy.special.ndtri(1 - p/2)
                if signif > 3:
                    print('******************************')
                print(colname1, colname2)
                print(orig_spearman, p, signif)
                if signif > 3:
                    print('******************************')
                print()

def spearman_helper(col1,col2,num_iters):
    col_len = len(col1)
    return np.array([scipy.stats.spearmanr(col1,np.random.choice(col2,size=col_len,replace=False))[0] for i in range(num_iters)])
    
