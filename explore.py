import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

N = 152 # total number of years
W = 5 # window for averaging earnings
print('total number of data points N = ', N)
print('averaging window W = ', W)
annualDF = pd.read_excel('data4.xlsx', sheet_name = 'data')
annual = annualDF.values
div = annual[:N, 1].astype(float) #annual dividends
earn = annual[:N, 2].astype(float) #annual earnings
index = annual[:N + 1, 3].astype(float) #annual index values
cpi = annual[:N + 1, 4].astype(float) #annual consumer price index
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) # total real returns
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # real wealth
plt.plot(range(1871, 1871 + N + 1), rindex)
plt.yscale('log')
plt.show()
plt.plot(range(1871, 1871 + N + 1), rwealth)
plt.yscale('log')
plt.show()
cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)]
cumdiv = [sum(rdiv[k:k+W])/W for k in range(N-W+1)]
plt.plot(range(1871, 1871 + N), rdiv)
plt.plot(range(1871, 1871 + N), rearn)
plt.yscale('log')
plt.show()
plt.plot(range(1870 + W, 1871 + N), cumdiv)
plt.plot(range(1870 + W, 1871 + N), cumearn)
plt.yscale('log')
plt.show()
PE = index[1:]/earn
PD = index[1:]/div
print('mean PE & PD = ', np.mean(PE), np.mean(PD))
CAPE = rindex[W:]/cumearn
CAPD = rindex[W:]/cumdiv
print('mean CAPE & CAPD = ', np.mean(CAPE), np.mean(CAPD))
plt.plot(range(1871, 1871 + N), PE)
plt.plot(range(1871, 1871 + N), PD)
plt.show()
plt.plot(range(1870 + W, 1871 + N), CAPE)
plt.plot(range(1870 + W, 1871 + N), CAPD)
plt.show()
