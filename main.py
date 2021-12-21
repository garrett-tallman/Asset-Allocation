import pandas as pd
import numpy as np
import yfinance as yf
from cvxpy import *

# read monthly prices from yfinance
mr = pd.DataFrame()
data = pd.DataFrame()
tickers = "SPY, AAPL, MSFT, GOOG, AMZN, TSLA, FB, NVDA, JPM, UNH, V, JNJ, HD, WMT, PG, BAC, MA, PFE, DIS, ADBE, ORCL"
data = yf.download(tickers=tickers, period="2y", interval="1mo")
data = data['Adj Close']
data.dropna(subset=["SPY"], inplace=True)
data = data[:-1]

# compute monthly returns
for s in data.columns:
    date = data.index[0]
    pr0 = data[s][date]
    for t in range(1, len(data.index)):
        date = data.index[t]
        pr1 = data[s][date]
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
req_return = 0.08
portfolio_value = 100
ret = r.T @ x
risk = quad_form(x, C)

# solve problem and write solution
while True:
    try:
        prob = Problem(Minimize(risk), [sum(x) == 1, ret >= req_return, x >= 0])
        prob.solve()
        print("----------------------")
        print("Optimal Portfolio")
        print("----------------------")
        for s in range(len(symbols)):
            print('[%s] = %0.2f' % (symbols[s], x.value[s] * 100) + '%')
        print("----------------------")
        for s in range(len(symbols)):
            print('[%s] = $%0.2f' % (symbols[s], x.value[s] * portfolio_value))
        print("----------------------")
        print('Exp ret = %0.2f' % ret.value)
        print('risk    = %0.3f' % (risk.value ** 0.5))
        print("----------------------")
    except:
        if req_return != 0:
            req_return = req_return - 0.001
            continue
        else:
            print('Error')
            break
    break
