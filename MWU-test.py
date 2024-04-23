import os
import numpy as np
import pandas as pd
import sys
from scipy.stats import mannwhitneyu
from scipy.stats import permutation_test
from scipy.stats import ttest_ind

np.seterr(invalid='ignore')

ExpCRM=np.loadtxt(sys.argv[1],usecols =np.concatenate((np.arange(6,113),np.arange(116,223))))
if ExpCRM.ndim == 1:
    ExpCRM = np.reshape(ExpCRM, (1, 214))
pvalue=[]
for record in ExpCRM:
    Exp=record[0:107]
    Prob=record[107:214]
    Exp_0=Exp[Prob<0.5]
    Exp_1=Exp[Prob>=0.5]
    #Exp_1_jittered=Exp_1+np.random.uniform(low=-0.00001, high=0.00001, size=len(Exp_1))
    if sys.argv[2] == "MWUtest":
        if sys.argv[4] == "enhancer":
            statistic, p_value = mannwhitneyu(Exp_0,Exp_1, alternative='less')
        else:
            statistic, p_value = mannwhitneyu(Exp_0,Exp_1, alternative='greater')
    if sys.argv[2] == "ttest":
        statistic, p_value = ttest_ind(Exp_0,Exp_1_jittered, alternative='less')
    if sys.argv[2] == "permutation_test_mean":
        def statistic(x, y):
            return np.mean(x) - np.mean(y)
        result = permutation_test((Exp_0,Exp_1_jittered),statistic, n_resamples=1000, alternative='less')
        p_value=result.pvalue
    if sys.argv[2] == "permutation_test_median":
        def statistic(x, y):
            return np.median(x) - np.median(y)
        result = permutation_test((Exp_0,Exp_1_jittered),statistic, n_resamples=1000, alternative='less')
        p_value=result.pvalue
    pvalue.append(p_value)

df = pd.DataFrame(pvalue)
df.to_csv("Pvalue_{}_{}_{}_nojitter.csv".format(sys.argv[2],sys.argv[3],sys.argv[4]), index=False)
