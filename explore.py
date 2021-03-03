import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

N = 149 #total number of years
dataDF = pd.read_excel('data7.xlsx', sheet_name = 'data')
data = dataDF.values
div = data[:N, 1].astype(float) #annual dividends
earn = data[:N, 2].astype(float) #annual earnings
index = data[:N + 1, 3].astype(float) #annual index values
cpi = data[:N + 1, 4].astype(float) #annual consumer price index
INIT_YEAR = 1871
LAST_YEAR = INIT_YEAR + N
print('Years', INIT_YEAR, LAST_YEAR)
W = 10 #window of averaging earnings
print('Window = ', W)

#Real values of dividends, earnings, and index    
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
#Next, compute trailing averaged earnings
cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)]

plt.figure(figsize=(7,6))
plt.plot(range(INIT_YEAR, LAST_YEAR), rdiv)
plt.plot(range(INIT_YEAR, LAST_YEAR), rearn)
plt.plot(range(INIT_YEAR + W - 1, LAST_YEAR), cumearn)
plt.legend(['dividends', 'earnings', 'avg earnings'], loc = 'lower right')
plt.yscale('log')
plt.show()

TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) #Total nominal return
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR)))
print('mean and stdev of annual total returns 1871-2020 = ', np.mean(TR), np.std(TR))
print('and that of annual total returns starting from cutoff = ', np.mean(TR[W:]), np.std(TR[W:]))

#Plot of real stock index
plt.figure(figsize=(7,6))
plt.plot(range(INIT_YEAR, LAST_YEAR + 1), rindex)
plt.yscale('log')
plt.show()

#Plot of inflation-adjusted wealth, including reinvested dividends
plt.figure(figsize=(7,6))
plt.plot(range(INIT_YEAR, LAST_YEAR + 1), rwealth)
plt.yscale('log')
plt.show()

rearngr = np.diff(np.log(cumearn)) #real earnings growth
print('Mean and stdev of averaged 10-year trailing earnings growth = ', np.mean(rearngr), np.std(rearngr))
CAPE = rindex[W:]/cumearn
print('mean CAPE = ', np.mean(CAPE))
print('Corr Shiller CAPE ratio and next year real return = ', stats.pearsonr(CAPE[:-1], TR[W:])[0])
print('Corr log CAPE ratio and next year real return = ', stats.pearsonr(np.log(CAPE)[:-1], TR[W:])[0])

#Finally, compute total return-adjusted trailing averaged earnings
TRearn = rearn*(rwealth[1:]/rindex[1:])
TRcumearn = [sum(TRearn[k:k+W])/W for k in range(N-W+1)]
TRCAPE = rwealth[W:]/TRcumearn
print('mean TR-CAPE = ', np.mean(TRCAPE))

#Comparison of Shiller CAPE and TR-CAPE
plt.figure(figsize=(7,6))
plt.plot(range(INIT_YEAR + W, LAST_YEAR + 1), CAPE)
plt.plot(range(INIT_YEAR + W, LAST_YEAR + 1), TRCAPE)
plt.legend(['CAPE', 'TR-CAPE'], loc = 'lower right')
plt.title('Shiller CAPE and TR-adjusted CAPE ratios')
plt.show()

print('Corr TR CAPE ratio and next year real return = ', stats.pearsonr(TRCAPE[:-1], TR[W:])[0])
print('Corr log TR CAPE ratio and next year real return = ', stats.pearsonr(np.log(TRCAPE)[:-1], TR[W:])[0])
TRearngr = np.diff(np.log(TRcumearn))
print('Mean and stdev of TR-adjusted averaged 10-year trailing earnings growth = ', np.mean(TRearngr), np.std(TRearngr))
print('current CAPE and TR-CAPE = ', CAPE[-1], TRCAPE[-1])

#correlation between our log valuation measures and 10-year future total returns
futureTR = [np.mean(TR[W + k:2 * W + k - 1]) for k in range(N - 2*W + 1)]
print('correlation between log TR-CAPE and future 10-year returns = ', stats.pearsonr(np.log(TRCAPE[:-W]), futureTR)[0])
print('correlation between log CAPE and future 10-year returns = ', stats.pearsonr(np.log(CAPE[:-W]), futureTR)[0])

#correlation between original CAPE and TR-CAPE
print('correlation between log CAPE and log TR-CAPE = ', stats.pearsonr(np.log(TRCAPE), np.log(CAPE))[0])

print('correlation between log TR-CAPE and future 1-year returns = ', stats.pearsonr(np.log(TRCAPE[:-1]), TR[W:])[0])
print('correlation between log CAPE and future 1-year returns = ', stats.pearsonr(np.log(CAPE[:-1]), TR[W:])[0])