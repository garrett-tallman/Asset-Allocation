import pandas as pd
import numpy as np
from cvxpy import *

# read monthly_prices.csv
mp = pd.read_csv("monthly_prices_homework.csv",index_col=0)
mr = pd.DataFrame()

# compute monthly returns
for s in mp.columns:
    date = mp.index[0]
    pr0 = mp[s][date]
    for t in range(1, len(mp.index)):
        date = mp.index[t]
        pr1 = mp[s][date]
        ret = (pr1 - pr0) / pr0
        mr.at[date, s] = ret
        pr0 = pr1

# get symbol names
symbols = mr.columns

# convert monthly return date fame to a numpy matrix
return_data = mr.values.T

# compute mean return
r = np.asarray(np.mean(return_data, axis=1))

# covariance
C = np.asmatrix(np.cov(return_data))

# print out expected return and std deviation
print("----------------------")
for j in range(len(symbols)):
    print('%s: Exp ret = %f, Risk = %f' % (symbols[j], r[j], C[j, j] ** 0.5))

# set up optimization model
n = len(symbols)
x = Variable(n)
req_return = 0.02
ret = r.T @ x
risk = quad_form(x, C)
prob = Problem(Minimize(risk), [sum(x) == 1, ret >= req_return, x >= 0])

# solve problem and write solution
try:
    prob.solve()
    print("----------------------")
    print("Optimal Portfolio")
    print("----------------------")
    for s in range(len(symbols)):
        print('[%s] = %f' % (symbols[s], x.value[s]))
    print("----------------------")
    for s in range(len(symbols)):
        print('[%s] = $%f' % (symbols[s], x.value[s]*1000))
    print("----------------------")
    print('Exp ret = %f' % ret.value)
    print('risk    = %f' % (risk.value ** 0.5))
    print("----------------------")
except:
    print('Error')