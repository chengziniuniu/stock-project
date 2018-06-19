################# Force Index ########################################################

# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data
# 买方力量指标 Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data


# 收集AAPL的数据
data = data.DataReader('AAPL',data_source='yahoo',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)

# Compute 
n = 1
AAPL_ForceIndex = ForceIndex(data,n)
print(AAPL_ForceIndex)