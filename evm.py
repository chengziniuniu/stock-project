# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
 
# 简易波动指标 Ease of Movement 
def EVM(data, ndays): 
 dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
 br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
 EVM = dm / br 
 EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM') 
 data = data.join(EVM_MA) 
 return data 

data = data.DataReader('AAPL',data_source='yahoo',start='1/1/2015', end='1/1/2016')
data = pd.DataFrame(data)

# 基于十四天移动平局计算 简易波动指标
n = 14
AAPL_EVM = EVM(data, n)
EVM = AAPL_EVM['EVM']

#画图
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('AAPL Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(EVM,'k',lw=0.75,linestyle='-',label='EVM(14)')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('EVM values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)