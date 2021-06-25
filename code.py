import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

np.random.seed(25)
N = 150 #total number of years
dataDF = pd.read_excel('data7.xlsx', sheet_name = 'data')
data = dataDF.values
div = data[:N, 1].astype(float) #annual dividends
earn = data[:N, 2].astype(float) #annual earnings
index = data[:N + 1, 3].astype(float) #annual index values
cpi = data[:N + 1, 4].astype(float) #annual consumer price index

INIT = 1871 #initial year
W = 5 #window of averaging earnings
T = N - W + 1
NEW = INIT + W #initial year after taking window, 1876
LAST = INIT + N + 1 #last year, 2021
print('Years', INIT, LAST)
print('Window = ', W)
#Real values of dividends, earnings, and index    
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
#Next, compute trailing averaged earnings
cumearn = [sum(rearn[k:k+W])/W for k in range(T)]
plt.figure(figsize=(7,6))
plt.plot(range(INIT, LAST - 1), rdiv)
plt.plot(range(INIT, LAST - 1), rearn)
plt.plot(range(NEW, LAST), cumearn)
plt.legend(['dividends', 'earnings', 'avg earnings'], loc = 'lower right')
plt.yscale('log')
plt.show()
TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) #Total nominal return
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR)))
print('mean and stdev of annual total returns 1871-2020 = ', np.mean(TR), np.std(TR))
print('and that of annual total returns starting from cutoff = ', np.mean(TR[W:]), np.std(TR[W:]))

#Plot of real stock index
plt.figure(figsize=(7,6))
plt.plot(range(INIT, LAST), rindex)
plt.yscale('log')
plt.show()

#Plot of inflation-adjusted wealth, including reinvested dividends
plt.figure(figsize=(7,6))
plt.plot(range(INIT, LAST), rwealth)
plt.yscale('log')
plt.show()

growth = np.diff(np.log(cumearn)) #real earnings growth
CAPE = rindex[W:]/cumearn
print('mean CAPE = ', np.mean(CAPE))
print('Corr Shiller CAPE ratio and next year real return = ', stats.pearsonr(CAPE[:-1], TR[W:]))
print('Corr log CAPE ratio and next year real return = ', stats.pearsonr(np.log(CAPE)[:-1], TR[W:]))

#Finally, compute total return-adjusted trailing averaged earnings
TRearn = rearn*(rwealth[1:]/rindex[1:])
TRcumearn = [sum(TRearn[k:k+W])/W for k in range(N-W+1)]
TRCAPE = rwealth[W:]/TRcumearn
print('mean TR-CAPE = ', np.mean(TRCAPE))

#Comparison of Shiller CAPE and TR-CAPE
plt.figure(figsize=(7,6))
plt.plot(range(NEW, LAST), CAPE)
plt.plot(range(NEW, LAST), TRCAPE)
plt.legend(['CAPE', 'TR-CAPE'], loc = 'lower right')
plt.title('Shiller CAPE and TR-adjusted CAPE ratios')
plt.show()

#mean and stdev for total real returns and real earnings growth
print('mean of total real returns = ', np.mean(TR[W:]))
print('mean real earnings growth = ', np.mean(growth))
print('stdev of total real returns = ', np.std(TR[W:]))
print('stdev of real earnings growth = ', np.std(growth))

IDY = TR[W:] - growth #implied dividend yield
# cumulative implied dividend yield, after detrending it becomes heat measure
cumIDY = np.append(np.array([0]), np.cumsum(IDY))

# graphs of ACF and QQ for real earnings growth terms
plot_acf(growth)
plt.show()
qqplot(growth, line = 's')
plt.show()

# main regression
DF = pd.DataFrame({'const' : 1, 'trend' : range(T-1), 'Bubble' : cumIDY[:-1]})
Regression = OLS(IDY, DF).fit()
print(Regression.summary())
coefficients = Regression.params
intercept = coefficients[0]
trendCoeff = coefficients[1]
heatCoeff = coefficients[2]
avgIDY = trendCoeff/abs(heatCoeff)
print('avgIDY = ', avgIDY)
avgHeat = (intercept - avgIDY)/abs(heatCoeff)
print('long-term average heat measure = ', avgHeat)

Heat = cumIDY - avgIDY * range(T) #Heat measure
plt.figure(figsize=(7,6))
plt.plot(range(NEW, LAST), Heat)
print('current heat measure = ', Heat[-1])
plt.title('Heat measure')
plt.show()
print('Correlation of heat measure and total returns = ', stats.pearsonr(Heat[:-1], TR[W:])[0])
residuals = IDY - Regression.predict(DF)

#analysis of regression residuals for white noise and normality
stderr = np.std(residuals)
print('stderr = ', stderr)
print('Shapiro-Wilk normality test for residuals', stats.shapiro(residuals))
print('Jarque-Bera normality test for residuals', stats.jarque_bera(residuals))
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
acf_resid, stat_resid, p_resid = stattools.acf(residuals, unbiased = True, fft=False, qstat = True)
acf_aresid, stat_aresid, p_aresid = stattools.acf(aresiduals, unbiased = True, fft=False, qstat = True)
print('Ljung-Box p-value for residuals, original values')
print('lag 5 = ', p_resid[4])
print('lag 10 = ', p_resid[9])
print('Ljung-Box p-value for residuals, absolute values')
print('lag 5 = ', p_aresid[4])
print('lag 10 = ', p_aresid[9])

# Checking dependence of regression residuals
print('Correlation between residuals and earnings growth')
print('Pearson, original values', stats.pearsonr(residuals, growth))
print('Spearman, original values', stats.spearmanr(residuals, growth))

# comparison of logarithm CAPE and the heat measure
logCAPE = np.log(CAPE)
lCAPE = logCAPE - np.ones(T)*logCAPE[0]
plt.plot(range(NEW, LAST), lCAPE)
plt.plot(range(NEW, LAST), Heat)
plt.legend(['CAPE', 'Heat'], loc = 'lower right')
plt.show()

#bootstrap for ruin probability
NSIMS = 10000
WINDOW = 25
times = np.random.choice(range(N - W - WINDOW + 1), NSIMS)
print('time horizon = ', WINDOW)

#simulation of individual autoregression of order one
def ARsim(initHeat):
    heat = [initHeat]
    innovations = np.random.normal(0, 1, WINDOW)
    for t in range(WINDOW):
        newHeat = avgHeat + (1 + heatCoeff)*(heat[t] - avgHeat) + stderr * innovations[t]
        heat.append(newHeat)
    return np.array(heat)
        
#ruin probability with withdrawal rate and initial heat
def Ruin(initHeat, rate):
    ruin = []
    for sim in range(NSIMS):
        heat = ARsim(initHeat)
        bootstrap = growth[times[sim]:times[sim] + WINDOW]
        wealth = 1
        t = 0
        while ((t < WINDOW) and (wealth > 0)):
            wealth = wealth * np.exp(bootstrap[t] + heat[t+1] - heat[t] + avgIDY) - rate
            t = t + 1
        if wealth < 0:
            ruin.append(1)
        if wealth > 0:
            ruin.append(0)
    return np.mean(ruin)

print('withdrawal rate 4%')
print('current heat =', Ruin(Heat[-1], 0.04))
print('long-term heat =', Ruin(avgHeat, 0.04))
print('withdrawal rate 5%')
print('current heat =', Ruin(Heat[-1], 0.05))
print('long-term heat =', Ruin(avgHeat, 0.05))
print('withdrawal rate 6%')
print('current heat =', Ruin(Heat[-1], 0.06))
print('long-term heat =', Ruin(avgHeat, 0.06))