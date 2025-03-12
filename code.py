import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

W = 5 # window for averaging earnings
print('averaging window W =', W)
annualDF = pd.read_excel('data4.xlsx', sheet_name = 'data')
annual = annualDF.values

div = annual[:-1, 1].astype(float) #annual dividends
earn = annual[:-1, 2].astype(float) #annual earnings
N = len(div) # total number of years
print('total number of data points N =', N)
index = annual[:, 3].astype(float) #annual index values
cpi = annual[:, 4].astype(float) #annual consumer price index

rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
TR = np.array([np.log(rdiv[k] + rindex[k+1]) - np.log(rindex[k]) for k in range(N)]) # total real returns
rwealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # real wealth

plt.plot(range(1871, 1871 + N + 1), rindex)
plt.xlabel('Years')
plt.ylabel('Log Index')
plt.title('Standard & Poor 500 Index Level')
plt.yscale('log')
plt.savefig('index.png', bbox_inches='tight')
plt.close()

plt.plot(range(1871, 1871 + N + 1), rwealth)
plt.xlabel('Years')
plt.ylabel('Log Wealth')
plt.title('Wealth Invested in Standard & Poor 500')
plt.yscale('log')
plt.savefig('wealth.png', bbox_inches='tight')
plt.close()

cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)]
cumdiv = [sum(rdiv[k:k+W])/W for k in range(N-W+1)]
plt.plot(range(1871, 1871 + N), rdiv, label = 'Dividends')
plt.plot(range(1871, 1871 + N), rearn, label = 'Earnings')
plt.xlabel('Years')
plt.ylabel('Earnings and Dividends')
plt.title('Log Earnings and Dividends for S&P 500')
plt.yscale('log')
plt.legend(bbox_to_anchor=(0.1, 0.9), loc='upper left', prop={'size': 12})
plt.savefig('fundamentals.png', bbox_inches='tight')
plt.close()

plt.plot(range(1870 + W, 1871 + N), cumdiv, label = 'Dividends')
plt.plot(range(1870 + W, 1871 + N), cumearn, label = 'Earnings')
plt.xlabel('Years')
plt.ylabel('Earnings and Dividends')
plt.title('Trailing Averaged 5-year Earnings and Dividends for S&P 500')
plt.yscale('log')
plt.legend(bbox_to_anchor=(0.1, 0.9), loc='upper left', prop={'size': 12})
plt.savefig('trailing-fundamentals.png', bbox_inches='tight')
plt.close()

PE = index[1:]/earn
PD = index[1:]/div
print('mean PE & PD = ', np.mean(PE), np.mean(PD))
CAPE = rindex[W:]/cumearn
CAPD = rindex[W:]/cumdiv
print('mean CAPE & CAPD = ', np.mean(CAPE), np.mean(CAPD))

plt.plot(range(1871, 1871 + N), PE, label = 'Price-Earnings')
plt.plot(range(1871, 1871 + N), PD, label = 'Price-Dividend')
plt.xlabel('Years')
plt.ylabel('Ratios')
plt.title('Price-Earnings and Price-Dividend Ratios for S&P 500')
plt.legend(bbox_to_anchor=(0.1, 0.9), loc='upper left', prop={'size': 12})
plt.savefig('classic-ratios.png', bbox_inches='tight')
plt.close()

plt.plot(range(1870 + W, 1871 + N), CAPE, label = 'Price-Earnings')
plt.plot(range(1870 + W, 1871 + N), CAPD, label = 'Price-Dividend')
plt.xlabel('Years')
plt.ylabel('Ratios')
plt.title('Cyclically Adjusted Ratios for S&P 500')
plt.legend(bbox_to_anchor=(0.1, 0.9), loc='upper left', prop={'size': 12})
plt.savefig('cyclical-ratios.png', bbox_inches='tight')
plt.close()

payout = [cumdiv[k]/cumearn[k] for k in range(N - W + 1)]
plt.plot(range(1870 + W, 1871 + N), payout)
plt.xlabel('Years')
plt.ylabel('Payout Ratio')
plt.title('S&P 500 Payout Ratio')
plt.savefig('payout.png', bbox_inches='tight')
plt.close()

print('mean payout = ', np.mean(payout))
print('mean growth of annual dividends = ', (np.log(rdiv[-1]) - np.log(rdiv[0]))/(N-1))
print('mean growth of averaged trailing dividends = ', (np.log(cumdiv[-1]) - np.log(cumdiv[0]))/(N-W))
print('mean growth of annual earnings = ', (np.log(rearn[-1]) - np.log(rearn[0]))/(N-1))
print('mean growth of averaged trailing earnings = ', (np.log(cumearn[-1]) - np.log(cumearn[0]))/(N-W))

NFUNDAMENTALS = 4
fundamentals = [rearn, cumearn, rdiv, cumdiv]
lags = [1, W, 1, W]
labels = ['1-Year Earnings', '5-Year Earnings', '1-Year Dividends', '5-Year Dividends']

for k in range(NFUNDAMENTALS):
    fund = fundamentals[k]
    lag = lags[k]
    label = labels[k]
    print(label)
    growth = np.diff(np.log(fund))
    print('mean and stdev of total returns = ', np.mean(TR[lag-1:]), np.std(TR[lag-1:]))
    print('stdev of growth = ', np.std(growth))
    IDY = TR[lag:] - growth
    cumIDY = np.append(np.array([0]), np.cumsum(IDY))
    
    # main regression
    DF = pd.DataFrame({'const' : 1, 'trend' : range(N - lag), 'bubble' : cumIDY[:-1]})
    Regression = OLS(IDY, DF).fit()
    print('Main Regression when fundamentals are ' + label)
    print(Regression.summary())
    coefficients = Regression.params
    intercept = coefficients['const']
    trend_coeff = coefficients['trend']
    bubble_coeff = coefficients['bubble']
    avgIDY = trend_coeff/abs(bubble_coeff)
    print('avgIDY = ', avgIDY)
    avg_bubble_measure = (intercept - avgIDY)/abs(bubble_coeff)
    print('long-term average bubble = ', avg_bubble_measure)

    # Computation of the bubble measure
    Bubble = cumIDY - avgIDY * range(N - lag + 1)
    #plot of the new bubble measure
    plt.figure(figsize=(7,6))
    plt.plot(range(1870 + lag, 1871 + N), Bubble)
    print('current valuation measure = ', Bubble[-1])
    plt.xlabel('Years')
    plt.ylabel('Valuation Measure')
    plt.title('New Standard & Poor 500 Valuation Measure ' + label)
    plt.savefig(label + ' Valuation.png')
    plt.close()

    print('Correlation of bubble measure and total returns = ', stats.pearsonr(Bubble[:-1], TR[lag:])[0])
    # computation and testing for regression residuals
    residuals = IDY - Regression.predict(DF)
    stderr = np.std(residuals)
    print('stderr = ', stderr)
    print('Shapiro-Wilk normality test for residuals', stats.shapiro(residuals)[1])
    print('Jarque-Bera normality test for residuals', stats.jarque_bera(residuals)[1])
    aresiduals = abs(residuals)
    qqplot(residuals, line = 's')
    plt.title('quantile-quantile plot of residuals ' + label)
    plt.savefig(label + ' QQ.png')
    plt.close()
    plot_acf(residuals, zero = False)
    plt.title('original values of residuals ' + label)
    plt.savefig(label + ' Original ACF.png')
    plt.close()
    plot_acf(aresiduals, zero = False)
    plt.title('absolute values of residuals ' + label)
    plt.savefig(label + ' Absolute ACF.png')
    plt.close()

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
    print('Pearson, original values', stats.pearsonr(residuals, growth))
    print('Spearman, original values', stats.spearmanr(residuals, growth))
    
    if label == '5-Year Earnings':
        plt.plot(range(1870 + lag, 1871 + N), np.log(CAPE) - np.log(CAPE[0])*np.ones(N+1-lag), label = 'Shiller CAPE')
        plt.plot(range(1870 + lag, 1871 + N), Bubble, label = 'New Measure')
        plt.xlabel('Years')
        plt.ylabel('Measure')
        plt.title('Shiller CAPE vs New Measure')
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', prop={'size': 12})
        plt.savefig('compare.png')
        plt.close()

    