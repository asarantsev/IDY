import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

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

cumdiv = [sum(rdiv[k:k+W])/W for k in range(N-W+1)] # averaged trailing dividends
divgrow = np.diff(np.log(cumdiv))
print('mean and stdev of TR = ', np.mean(TR[W:]), np.std(TR[W:]))
print('stdev of dividends growth = ', np.std(divgrow))
IDY = TR[W:] - divgrow
cumIDY = np.append(np.array([0]), np.cumsum(IDY))
CAPD = rindex[W:]/cumdiv
print('mean CAPD = ', np.mean(CAPD))
plt.plot(range(1870 + W, 1871 + N), CAPD)
plt.show()

# main regression
DF = pd.DataFrame({'const' : 1, 'trend' : range(N-W), 'Bubble' : cumIDY[:-1]})
Regression = OLS(IDY, DF).fit()
print(Regression.summary())
coefficients = Regression.params
intercept = coefficients[0]
trend_coeff = coefficients[1]
bubble_coeff = coefficients[2]
avgIDY = trend_coeff/abs(bubble_coeff)
print('avgIDY = ', avgIDY)
avg_bubble_measure = (intercept - avgIDY)/abs(bubble_coeff)
print('long-term average bubble = ', avg_bubble_measure)

# Computation of the bubble measure
Bubble = cumIDY - avgIDY * range(N - W + 1)

#plot of the new bubble measure
plt.figure(figsize=(7,6))
plt.plot(range(1870 + W, 1871 + N), Bubble)
print('current bubble measure = ', Bubble[-1])
plt.title('Bubble measure')
plt.show()

print('Correlation of bubble measure and total returns = ', stats.pearsonr(Bubble[:-1], TR[W:])[0])

# computation and testing for regression residuals
residuals = IDY - Regression.predict(DF)
stderr = np.std(residuals)
print('stderr = ', stderr)
print('Shapiro-Wilk normality test for residuals', stats.shapiro(residuals)[1])
print('Jarque-Bera normality test for residuals', stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('residuals')
plt.show()
plot_acf(residuals)
plt.title('original values of residuals')
plt.show()
plot_acf(aresiduals)
plt.title('absolute values of residuals')
plt.show()

# p-values for Ljung-Box test of residuals, original and absolute values
acf_resid, stat_resid, p_resid = stattools.acf(residuals, fft=False, qstat = True)
acf_aresid, stat_aresid, p_aresid = stattools.acf(aresiduals, fft=False, qstat = True)

print('Ljung-Box p-value for residuals, original values')
print('lag 5 = ', p_resid[4])
print('lag 10 = ', p_resid[9])

print('Ljung-Box p-value for residuals, absolute values')
print('lag 5 = ', p_aresid[4])
print('lag 10 = ', p_aresid[9])

print('Correlation between residuals and dividend growth')
print('Pearson, original values', stats.pearsonr(residuals, divgrow))
print('Spearman, original values', stats.spearmanr(residuals, divgrow))

plt.plot(range(1870 + W, 1871 + N), np.log(CAPD) - np.log(CAPD[0])*np.ones(N+1-W))
plt.plot(range(1870 + W, 1871 + N), Bubble)
plt.show()