################ Bollinger Band #############################

# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data, wb

# Compute the Bollinger Bands 
def BBANDS(data, ndays):
    MA = pd.Series(pd.rolling_mean(data['Close'], ndays)) 
    SD = pd.Series(pd.rolling_std(data['Close'], ndays))
    
    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper BollingerBand') 
    data = data.join(B1) 
     
    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower BollingerBand') 
    data = data.join(B2) 
     
    return data
 
# 从yahoo财经收集数据
data = data.DataReader('^NSEI',data_source='yahoo',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)

# 基于50日移动平局计算Bollinger Band
n = 50
NIFTY_BBANDS = BBANDS(data, n)
NIFTY_BBANDS.plot()