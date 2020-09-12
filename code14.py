import pandas as pd
import numpy as np
import statsmodels
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels import api


# Random Seed for reproducibility
np.random.seed(1234)
# number of years, 1871-2019
N = 149
annualDF = pd.read_excel('Total.xlsx', sheet_name = 'Annual')
annual = annualDF.values
div = annual[:N, 1] #annual dividends
earn = annual[:N, 2] #annual earnings
monthlyDF = pd.read_excel('Total.xlsx', sheet_name = 'Monthly')
monthly = monthlyDF.values 
index = monthly[::12, 1].astype(float) #annual index values
cpi = monthly[::12, 2].astype(float) #annual consumer price index
ldiv = np.log(div) #logarithmic dividends
learn = np.log(earn) #logarithmic earnings 
lindex = np.log(index) 
lcpi = np.log(cpi)

#Exploratory plots; data is plotted versus years
def exp_plot(data, years, ylabel_text, title_text, log_flag = True):
    plt.figure(figsize=(7,6))
    plt.plot(range(years[0], years[1]), data)
    plt.xlabel('Years', fontsize = 7)
    plt.ylabel(ylabel_text, fontsize = 7)
    plt.title(title_text, fontsize = 10)
    if log_flag == True:
        plt.yscale('log')
    plt.show()

#Real values of dividends, earnings, and index    
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
TR = np.array([np.log(div[k] + index[k+1]) - np.log(index[k]) for k in range(N)]) #Total nominal return
lwealth = np.append([0], np.cumsum(TR)) #logarithmic nominal wealth for 1$ invested in Jan 1871, dividends reinvested
wealth = np.exp(lwealth) #nominal wealth for 1$ invested in Jan 1871, dividends reinvested
earngr = np.diff(learn) #nominal earnings growth
infl = np.diff(lcpi) #logarithmic inflation rates
realret = TR - infl #total real return
lrwealth = lwealth - lcpi + lcpi[0]*np.ones(N+1) #logarithmic real wealth, 1$ = Jan 1871
rwealth = np.exp(lrwealth) #real wealth
rearngr = earngr - infl[1:] #real earnings growth
#Now we can just call the function with all the exploratory plots
exp_plot(rdiv, (1871, 2020), 'Dividends', 'Log plot of real dividends', True)
exp_plot(rearn, (1871, 2020), 'Earnings', 'Log plot of real earnings', True)
exp_plot(rindex, (1871, 2021), 'Index', 'Log plot of real index', True)
exp_plot(cpi, (1871,2021), 'CPI', 'Log plot of CPI')
exp_plot(rwealth, (1871, 2021), 'Real wealth, Jan.1871 = $1', 'Log plot of real wealth', True)

W = 10 #window of averaging earnings
print('Window = ', W)
print('Average return = ', np.mean(realret[W+1:]))
cumearn = [sum(rearn[k:k+W])/W for k in range(N-W+1)]
rearngr = np.array([np.log(cumearn[k+1]/cumearn[k]) for k in range(N-W)])
idivyield = realret[W:] - rearngr #implied dividend yield
exp_plot(cumearn, (1870 + W, 2020), 'Earnings', 'Log plot of real 10-year trailing earnings', True)
plt.hist(rearngr, bins = 20)
plt.title('Histogram of 148 real earnings growth data')
plt.show()
qqplot(rearngr, line = 's')
plt.title('QQ plot for real earnings growth')
plt.show()
plot_acf(rearngr)
plt.show()
plot_acf(abs(rearngr))
plt.show()
print('Shapiro-Wilk for real earnings growth p = ', stats.shapiro(rearngr)[1])
print('Jarque-Bera for real earnings growth p =', stats.jarque_bera(rearngr)[1])

#p-values for Ljung-Box test
autocorr, stat, p = stattools.acf(rearngr, unbiased = True, fft=False, qstat = True)
plt.plot(range(1, len(p)+1), p)
plt.xlabel('Lag')
plt.ylabel('p')
plt.title('Ljung-Box p-value for real earnings growth')
plt.show()

ACVFrearngr = stattools.acovf(rearngr, fft = False)
acovREG = ACVFrearngr[:W]
eststdREG = np.sqrt((acovREG[0] + 2*sum([(1 - k/W)*acovREG[k] for k in range(1, W)]))/W)
print('estimated standard deviation for averaged real earnings growth = ', eststdREG)
#mean and stdev for real earnings growth
meanREG = np.mean(rearngr)
stdREG = np.std(rearngr)
print('Mean of real earnings growth = ', round(meanREG, 5))
print('Std of real earnings growth = ', round(stdREG, 5))

print('mean of exp = ', sum([np.exp(-item) for item in rearngr])/N)

Mean = np.mean(realret)
Stdev = np.std(realret)
print('Mean of total real return = ', round(Mean, 5))
print('Stdev of total real return = ', round(Stdev, 5))
# Make the plot
exp_plot(idivyield, (1870 + W, 2019), 'Implied dividend yield', 'Total return minus earnings growth', False)
#mean and stdev for implied dividend yield
meanidy = np.mean(idivyield)
stdidy = np.std(idivyield)
print('Annualized average of implied dividend yield = ', round(meanidy, 5))
print('Annualized stdev of implied dividend yield = ', round(stdidy, 5))

#bubble not yet detrended measure: cumulative sum of implied dividend yield terms
Bubble = np.append([0], np.cumsum(idivyield))
# Time to plot
exp_plot(Bubble, (1870 + W, 2020), 'Bubble measure', 'Cumulative implied dividend yield, not detrended', False)
#Regression factors: Design matrix
Y = pd.DataFrame({'Const': 1, 'Bubble': Bubble[:-1], 'Trend': range(N-W)})
#main regression fit
Reg = api.OLS(idivyield, Y).fit()
print(Reg.summary()) #Regression results
intercept = Reg.params[0] #alpha from article
bubble_measure_coeff = Reg.params[1] #minus beta from article
trend_coeff = Reg.params[2] #b = minus beta times c (=long-term implied dividend yield)

#Regression Residuals Analysis
Residuals = idivyield - Reg.predict(Y) #regression residuals
stderr = np.sqrt((1/(N-3-W))*np.dot(Residuals, Residuals)) #standard error for regression residuals
print('stderr = ', stderr)

IDY = trend_coeff/abs(bubble_measure_coeff) #long-term average of implied dividend yield c in article
print('Trend coefficient is ', round(IDY, 5))
detrend_bubble_measure = Bubble - IDY*range(N-W+1) #detrended bubble measure
exp_plot(detrend_bubble_measure, (1870 + W, 2020), 'detrended bubble measure', 'Time detrended bubble measure', False)
print('Current detrended bubble measure = ', round(detrend_bubble_measure[-1], 5))
print('Avg real earnings growth = ', round(meanREG, 5)) #g from article
long_term_bubble = (IDY - intercept)/bubble_measure_coeff #long-term bubble measure, h from article
print('Long-term bubble measure = ', round(long_term_bubble, 5))
current_bubble = detrend_bubble_measure[-1] #bubble measure as of January 2020
print('Current bubble measure = ', current_bubble)
limtrret = meanREG + IDY #long-run total real returns, c + g from article
print('Long-term total real return = ', round(limtrret, 5))

qqplot(Residuals, line = 's')
plt.show()
plot_acf(Residuals)
plt.show()
plot_acf(abs(Residuals))
plt.show()
print('Shapiro-Wilk p = ', stats.shapiro(Residuals)[1])
print('Pearson corr p = ', stats.pearsonr(Residuals, rearngr)[1], stats.pearsonr(abs(Residuals), abs(rearngr))[1])
print('Spearman corr p = ', stats.spearmanr(Residuals, rearngr)[1], stats.spearmanr(abs(Residuals), abs(rearngr))[1])
print('Kendall corr p = ', stats.kendalltau(Residuals, rearngr)[1], stats.kendalltau(abs(Residuals), abs(rearngr))[1])

#p-values for Ljung-Box test
autocorr_res, stat_res, p_res = stattools.acf(Residuals, unbiased = True, fft=False, qstat = True)
plt.plot(range(1, len(p_res)+1), p_res)
plt.xlabel('Lag')
plt.ylabel('p')
plt.title('Ljung-Box p-value for regression residuals')
plt.show()
autocorr_res, stat_res, p_res = stattools.acf(abs(Residuals), unbiased = True, fft=False, qstat = True)
plt.plot(range(1, len(p_res)+1), p_res)
plt.xlabel('Lag')
plt.ylabel('p')
plt.title('Ljung-Box p-value for absolute values of residuals')
plt.show()

def withdrawal(w, REG, bubble, T):
    wealth = [1]
    currBubble = bubble
    noise = np.random.normal(0, stderr, T)
    for t in range(T):
        nextIDY = bubble_measure_coeff * currBubble + intercept + noise[t]
        currBubble = currBubble + nextIDY - IDY
        currWealth = wealth[t] * np.exp(REG + nextIDY) * (1 - w)
        wealth.append(currWealth)
    return (min(wealth))

NSIMS = 10000
withdrawalRate = 0.04
T = 10
print('simulation with constant real earning growth and 4% withdrawal rate')
simResults = []
for sim in range(NSIMS):
    simResults.append(withdrawal(withdrawalRate, 0, current_bubble, T))
print('1% VaR minimal wealth, current, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.1))
simResults = []
for sim in range(NSIMS):
    simResults.append(withdrawal(withdrawalRate, 0.01, current_bubble, T))
print('1% VaR minimal wealth, current, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.1))
simResults = []
for sim in range(NSIMS):
    simResults.append(withdrawal(withdrawalRate, 0, long_term_bubble, T))
print('1% VaR minimal wealth, long term, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long term, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long term, REG 0%, withdrawal 4% = ', np.percentile(simResults, 0.1))
simResults = []
for sim in range(NSIMS):
    simResults.append(withdrawal(withdrawalRate, 0.01, long_term_bubble, T))
print('1% VaR minimal wealth, long term, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long term, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long term, REG 1%, withdrawal 4% = ', np.percentile(simResults, 0.1))

print('simulation with special withdrawal process')
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.01, 0, current_bubble, T))
print('1% VaR minimal wealth, current, w = 1%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, w = 1%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, w = 1%, ', np.percentile(simResults, 0.1))
print('simulation with special withdrawal process')
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.02, 0, current_bubble, T))
print('1% VaR minimal wealth, current, w = 2%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, w = 2%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, w = 2%, ', np.percentile(simResults, 0.1))
print('simulation with special withdrawal process')
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.03, 0, current_bubble, T))
print('1% VaR minimal wealth, current, w = 3%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, w = 3%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, w = 3%, ', np.percentile(simResults, 0.1))
print('simulation with special withdrawal process')
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.04, 0, current_bubble, T))
print('1% VaR minimal wealth, current, w = 4%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, current, w = 4%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, current, w = 4%, ', np.percentile(simResults, 0.1))
print('simulation with special withdrawal process')
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.01, 0, long_term_bubble, T))
print('1% VaR minimal wealth, long-term, w = 1%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long-term, w = 1%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long-term, w = 1%, ', np.percentile(simResults, 0.1))        
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.02, 0, long_term_bubble, T))
print('1% VaR minimal wealth, long-term, w = 2%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long-term, w = 2%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long-term, w = 2%, ', np.percentile(simResults, 0.1))        
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.03, 0, long_term_bubble, T))
print('1% VaR minimal wealth, long-term, w = 3%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long-term, w = 3%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long-term, w = 3%, ', np.percentile(simResults, 0.1))        
for sim in range(NSIMS):
    simResults.append(withdrawal(-0.04, 0, long_term_bubble, T))
print('1% VaR minimal wealth, long-term, w = 4%, ', np.percentile(simResults, 0.01))
print('5% VaR minimal wealth, long-term, w = 4%, ', np.percentile(simResults, 0.05))
print('10% VaR minimal wealth, long-term, w = 4%, ', np.percentile(simResults, 0.1))     
        
        
        
        
        
