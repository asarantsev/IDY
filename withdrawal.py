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

NSIMS = 1000
WINDOW = 10
NWINDOWS = 5
MWINDOW = NWINDOWS * WINDOW
RATES = np.arange(0.02, 0.16, 0.002)
times = np.random.choice(range(N - W - MWINDOW + 1), NSIMS)
errors = np.random.normal(0, stderr, MWINDOW * NSIMS)

def sim_bubble(initial_bubble):
    bubble_array = []
    bubble = [initial_bubble]
    for sim in range(NSIMS):
        for t in range(MWINDOW):
            new_bubble = avgBubble + (1 + bubble_coeff)*(bubble[t] - avgBubble) + errors[sim * MWINDOW + t]
            bubble.append(new_bubble)
        bubble_array.append(bubble)
    return np.array(bubble_array)
        
def withdrawal(initial_bubble, rate):
    print('rate =', round(rate, 3))
    bubble = sim_bubble(initial_bubble)
    Results = np.zeros(NWINDOWS)
    for sim in range(NSIMS):
        wealth = 1
        for k in range(NWINDOWS):
            if (wealth > 0):
                for t in range(WINDOW):
                    currt = k * WINDOW + t
                    if (wealth > 0):
                        wealth = wealth * np.exp(growth[currt + times[sim]] + bubble[sim, currt] - bubble[sim, currt] + avgIDY) - rate
                    else: 
                        Results[k] = Results[k] + 1
                        break
            else:
                Results[k] = Results[k] + 1
    return Results/NSIMS           

def Graph(init_bubble):
    finalRuin = []
    for rate in RATES:
        finalRuin.append(withdrawal(init_bubble, rate))
    finalRuin = np.array(finalRuin)
    for k in range(NWINDOWS):
        plt.plot(RATES, finalRuin[:, k])
    plt.show()
    return finalRuin

print(Graph(avgBubble))
print(Graph(Bubble[-1]))
    