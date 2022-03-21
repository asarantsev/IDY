import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

N = 151 #total number of years
FDATA = 0
LDATA = N
dataDF = pd.read_excel('data7.xlsx', sheet_name = 'data')
data = dataDF.values
div = data[FDATA:LDATA, 1].astype(float) #annual dividends
earn = data[FDATA:LDATA, 2].astype(float) #annual earnings
index = data[FDATA:LDATA + 1, 3].astype(float) #annual index values
cpi = data[FDATA:LDATA + 1, 4].astype(float) #annual consumer price index

FYEAR = 1871 + FDATA #initial year
W = 5 #window of averaging earnings
T = N - W + 1 #number of data points after taking the cutoff window
LYEAR = 2022 #last year
print('First and Last Years', FYEAR + W - 1, '--', LYEAR)
print('Averaging Window = ', W)
#Real values of dividends, earnings, and index    
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
#Next, compute trailing averaged earnings
cumearn = [sum(rearn[k:k+W])/W for k in range(T)]
plt.figure(figsize=(7,6))
plt.plot(range(FYEAR, LYEAR), rdiv)
plt.plot(range(FYEAR, LYEAR), rearn)
plt.plot(range(FYEAR + W - 1, LYEAR), cumearn)
plt.legend(['dividends', 'earnings', 'avg earnings'], loc = 'lower right')
plt.yscale('log')
plt.title('Annual Dividends and Earnings; Averaged Trailing Earnings')
plt.show()
TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) #Total nominal return
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR)))
print('mean and stdev of annual total returns = ', np.mean(TR[W:]), np.std(TR[W:]))

#Plot of real stock index
plt.figure(figsize=(7,6))
plt.plot(range(FYEAR, LYEAR+1), rindex)
plt.yscale('log')
plt.title('Inflation-Adjusted Stock Market Index')
plt.show()

#Plot of inflation-adjusted wealth, including reinvested dividends
plt.figure(figsize=(7,6))
plt.plot(range(FYEAR, LYEAR+1), rwealth)
plt.yscale('log')
plt.title('Inflation-Adjusted Stock Market Wealth')
plt.show()

growth = np.diff(np.log(cumearn)) #real earnings growth
CAPE = rindex[W:]/cumearn #CAPE

#Finally, compute total return-adjusted trailing averaged earnings
TRearn = rearn*(rwealth[1:]/rindex[1:])
TRcumearn = [sum(TRearn[k:k+W])/W for k in range(N-W+1)]
TRCAPE = rwealth[W:]/TRcumearn
print('mean TR-CAPE = ', np.mean(TRCAPE))

#Comparison of Shiller CAPE and TR-CAPE
plt.figure(figsize=(7,6))
plt.plot(range(FYEAR + W - 1, LYEAR), CAPE)
plt.plot(range(FYEAR + W - 1, LYEAR), TRCAPE)
plt.legend(['CAPE', 'TR-CAPE'], loc = 'lower right')
plt.title('Shiller CAPE and TR-adjusted CAPE ratios')
plt.show()

#mean and stdev for total real returns and real earnings growth
print('mean total real returns = ', np.mean(TR[W:]))
print('mean real earnings growth = ', np.mean(growth))
print('stdev total real returns = ', np.std(TR[W:]))
print('stdev real earnings growth = ', np.std(growth))

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
print('average difference between total real returns and real earnings growth = ', avgIDY)
avgHeat = (intercept - avgIDY)/abs(heatCoeff)
print('long-term average bubble measure = ', avgHeat)

Heat = cumIDY - avgIDY * range(T) #Heat measure
plt.figure(figsize=(7,6))
plt.plot(range(FYEAR + W - 1, LYEAR), Heat)
plt.title('bubble measure')
plt.show()
print('current bubble measure = ', Heat[-1])
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
plt.plot(range(FYEAR + W - 1, LYEAR), lCAPE)
plt.plot(range(FYEAR + W - 1, LYEAR), Heat)
plt.legend(['CAPE', 'Heat'], loc = 'lower right')
plt.show()

#bootstrap for ruin probability
NSIMS = 1000
WINDOW = 40
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

print('withdrawal rate 3%')
print('current heat =', Ruin(Heat[-1], 0.03))
print('long-term heat =', Ruin(avgHeat, 0.03))
print('withdrawal rate 4%')
print('current heat =', Ruin(Heat[-1], 0.04))
print('long-term heat =', Ruin(avgHeat, 0.04))
print('withdrawal rate 5%')
print('current heat =', Ruin(Heat[-1], 0.05))
print('long-term heat =', Ruin(avgHeat, 0.05))