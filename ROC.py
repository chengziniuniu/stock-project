import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# 动量指标Rate of Change (ROC)
def ROC(data,n):
     N = data['Close'].diff(n)
     D = data['Close'].shift(n)
     ROC = pd.Series(N/D,name='Rate of Change')
     data = data.join(ROC)
     return data 

data = data.DataReader('^NSEI',data_source='yahoo',start='6/1/2015',end='1/1/2016')
data = pd.DataFrame(data)

n = 5
NIFTY_ROC = ROC(data,n)
ROC = NIFTY_ROC['Rate of Change']

#画图
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('ROC values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)