# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

# 顺势指标Commodity Channel Index(CCI) 
def CCI(data, ndays): 
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),name = 'CCI') 
    data = data.join(CCI) 
    return data

# yahoo财经上收集nifty的数据:
data = data.DataReader('^NSEI',data_source='yahoo',start='1/1/2014', end='1/1/2016')
data = pd.DataFrame(data)

# 移动平均为20日的cci指标计算
n = 20
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']

# plot
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)