# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# 简单移动平均线
def SMA(data, ndays): 
     SMA = pd.Series(pd.rolling_mean(data['Close'], ndays), name = 'SMA') 
     data = data.join(SMA) 
     return data

# 指数加权移动平均 
def EWMA(data, ndays): 
     EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), 
     name = 'EWMA_' + str(ndays)) 
     data = data.join(EMA) 
     return data

data = data.DataReader('^NSEI',data_source='yahoo',start='1/1/2013', end='1/1/2016')
data = pd.DataFrame(data) 
close = data['Close']

# 简单移动平均
n = 50
SMA_NIFTY = SMA(data,n)
SMA_NIFTY = SMA_NIFTY.dropna()
SMA = SMA_NIFTY['SMA']

# 计算指数加权移动平均
ew = 200
EWMA_NIFTY = EWMA(data,ew)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA = EWMA_NIFTY['EWMA_200']

# 画图
plt.figure(figsize=(9,5))
plt.plot(data['Close'],lw=1, label='NSE Prices')
plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')
plt.plot(EWMA,'r', lw=1, label='200-day EWMA (red)')
plt.legend(loc=2,prop={'size':11})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)