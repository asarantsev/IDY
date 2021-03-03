import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools

N = 149 # overall number of years
W = 10 # window for averaging earnings
print('total number of data points N = ', N)
print('averaging window W = ', W)
annualDF = pd.read_excel('data7.xlsx', sheet_name = 'data')
annual = annualDF.values
div = annual[:N, 1].astype(float) # annual dividends
earn = annual[:N, 2].astype(float) # annual earnings
index = annual[:N + 1, 3].astype(float) # annual index values
cpi = annual[:N + 1, 4].astype(float) # annual consumer price index
rdiv = cpi[-1]*div/cpi[1:] # inflation-adjusted annual dividends
rearn = cpi[-1]*earn/cpi[1:] # inflation-adjusted annual earnings
rindex = cpi[-1]*index/cpi # inflation-adjusted annual index
TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) # total real returns
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # real wealth
TRearn = rearn*(rwealth[1:]/rindex[1:]) # TR-adjusted earnings
TRcumearn = [sum(TRearn[k:k+W])/W for k in range(N-W+1)] # TR-adjusted trailing earnings
TRCAPE = rwealth[W:]/TRcumearn # TR adjusted CAPE
value = np.log(TRCAPE) # log of TR adjusted CAPE
plt.plot(range(1871+W, 2021), value)
plt.title('Logarithm of TR-adjusted Shiller CAPE')
plt.show()
print('initial analysis of dependence of returns upon TR-CAPE and its log')
print('corr TR-adjusted CAPE and TR = ', round(stats.pearsonr(TRCAPE[:-1], TR[W:])[0], 5))
print('corr log TR-adjusted CAPE and TR = ', round(stats.pearsonr(value[:-1], TR[W:])[0], 5))

# Main regression
Regression = stats.linregress(value[:-1], value[1:])
intercept = Regression.intercept
slope = Regression.slope
print('AR(1) for log of TR-adjusted CAPE')
print('intercept and slope = ', round(intercept, 5), round(slope, 5))

# Analysis of residuals
residuals = value[1:] - slope * value[:-1] - intercept * np.ones(N-W)
aresiduals = abs(residuals)
print('stderr = ', round(np.std(residuals), 5))
print('Shapiro-Wilk normality test for residuals = ', stats.shapiro(residuals))
print('Jarque-Bera normality test for residuals = ', stats.jarque_bera(residuals))

qqplot(residuals, line = 's')
plt.title('residuals')
plt.show()

plot_acf(residuals)
plt.title('original residuals')
plt.show()

plot_acf(aresiduals)
plt.title('absolute residuals')
plt.show()

#p-values for Ljung-Box test of residuals, original and absolute values
acf_resid, stat_resid, p_resid = stattools.acf(residuals, unbiased = True, fft=False, qstat = True)
acf_aresid, stat_aresid, p_aresid = stattools.acf(aresiduals, unbiased = True, fft=False, qstat = True)

print('Ljung-Box p-value for residuals, original values')
print('lag 5 = ', p_resid[4])
print('lag 10 = ', p_resid[9])
print('lag 15 = ', p_resid[14])
print('lag 20 = ', p_resid[19])

print('Ljung-Box p-value for residuals, absolute values')
print('lag 5 = ', p_aresid[4])
print('lag 10 = ', p_aresid[9])
print('lag 15 = ', p_aresid[14])
print('lag 20 = ', p_aresid[19])

# growth of TR-adjusted trailing earnings
growth = np.diff(np.log(TRcumearn))
agrowth = abs(growth)

plot_acf(growth)
plt.title('Trailing earnings growth, original values')
plt.show()

plot_acf(agrowth)
plt.title('Trailing earnings growth, absolute values')
plt.show()

print('std growth = ', np.std(growth))
print('std TR = ', np.std(TR[W:]))

#Checking dependence of regression residuals
print('Correlation between residuals and earnings growth')
print('Pearson, original values', stats.pearsonr(residuals, growth)[1])
print('Pearson, absolute values', stats.pearsonr(aresiduals, agrowth)[1])
print('Spearman, original values', stats.spearmanr(residuals, growth)[1])
print('Spearman, absolute values', stats.spearmanr(aresiduals, agrowth)[1])
print('Kendall, original values', stats.kendalltau(residuals, growth)[1])
print('Kendall, absolute values', stats.kendalltau(aresiduals, agrowth)[1])

#Unit root test
print('Augmented Dickey-Fuller test p = ', stattools.adfuller(value)[1])