import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

N = 150 # total number of years
W = 10 # window for averaging earnings, change to 5, 10, 15, 20
print('total number of data points N =', N)
print('averaging window W =', W)
annualDF = pd.read_excel('data7.xlsx', sheet_name = 'data')
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
cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)] # averaged trailing earnings
growth = np.diff(np.log(cumearn))
mgrowth = np.mean(growth)
sgrowth = np.std(growth)
IDY = TR[W:] - growth
cumIDY = np.append(np.array([0]), np.cumsum(IDY))

# main regression
DF = pd.DataFrame({'const' : 1, 'trend' : range(N-W), 'Bubble' : cumIDY[:-1]})
Regression = OLS(IDY, DF).fit()
coefficients = Regression.params
intercept = coefficients[0]
trend_coeff = coefficients[1]
bubble_coeff = coefficients[2]
avgIDY = trend_coeff/abs(bubble_coeff)
avgBubble = (intercept - avgIDY)/abs(bubble_coeff)
Bubble = cumIDY - avgIDY * range(N-W+1)
residuals = IDY - Regression.predict(DF)
stderr = np.std(residuals)
bill = annual[:N, 6].astype(float)
bret = np.array([np.log(1 + bill[k + W]/100) - np.log(cpi[k + W]/cpi[k])/W for k in range(N-W)])
#plt.plot(range(1871 + W, 1871 + N), bret)
#plt.show()

def portfolio(gamma, initial_bubble, window, rate):
    times = np.random.choice(range(N - W - window + 1))
    bubble = initial_bubble
    errors = np.random.normal(0, stderr, window)
    wealth = 1
    for t in range(window):
        port = 1/(2*gamma) + (avgIDY + mgrowth + bubble_coeff * (bubble - avgBubble) - bret[t + times])/(gamma*(stderr**2 + sgrowth**2))
        #print(port)
        if wealth > 0:
            new_bubble = avgBubble + (1 + bubble_coeff)*(bubble - avgBubble) + errors[t]
            pRet = port * np.exp(growth[t + times] + new_bubble - bubble + avgIDY) + (1 - port) * np.exp(bret[t + times])
            wealth = wealth * pRet - rate
            bubble = new_bubble
        else:
            wealth = 0
            break
    return wealth

def portfolioAbs(gamma, initial_bubble, window, rate):
    times = np.random.choice(range(N - W - window + 1))
    bubble = initial_bubble
    errors = np.random.normal(0, stderr, window)
    wealth = 1
    for t in range(window):
        port = 1/(2*gamma) + (avgIDY + mgrowth + bubble_coeff * (bubble - avgBubble) - bret[t + times])/(gamma*(stderr**2 + sgrowth**2))
        if (port > 1):
            port = 1
        if (port < 0):
            port = 0
        if wealth > 0:
            new_bubble = avgBubble + (1 + bubble_coeff)*(bubble - avgBubble) + errors[t]
            pRet = port * np.exp(growth[t + times] + new_bubble - bubble + avgIDY) + (1 - port) * np.exp(bret[t + times])
            wealth = wealth * pRet - rate
            bubble = new_bubble
        else:
            wealth = 0
            break
    return wealth

NSIMS = 1000
RATE = 0.05
final = []
for rate in np.arange(0.1, 5, 0.1):
    results = []
    for sim in range(NSIMS):
        results.append(portfolioAbs(rate, avgBubble, 30, RATE))
    final.append(1 - np.count_nonzero(results)/NSIMS)

plt.plot(np.arange(0.1, 5, 0.1), final)
plt.show()

final = []
for rate in np.arange(0.1, 5, 0.1):
    results = []
    for sim in range(NSIMS):
        results.append(portfolioAbs(rate, avgBubble, 50, RATE))
    final.append(1 - np.count_nonzero(results)/NSIMS)

plt.plot(np.arange(0.1, 5, 0.1), final)
plt.show()