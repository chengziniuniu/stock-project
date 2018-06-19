#!/usr/bin/python
# -*- coding: utf-8 -*-

# grid_search.py

from __future__ import print_function

import numpy as np
import pandas as pd
import datetime

import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from pandas_datareader import data

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    创建一个DataFrame存储当天数据和滞后的数据，数据包括调整后的收益率和交易量
    """

    ts = data.DataReader(
    	symbol, "yahoo", 
    	start_date-datetime.timedelta(days=365), 
    	end_date
    )

    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    for i in range(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    #为了保证QDA模型的顺利运行 将收益为零的点付一个很小的值
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = \
        tslag["Lag%s" % str(i+1)].pct_change()*100.0

    #创建一个Direction栏 表示上涨和下跌
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


if __name__ == "__main__":
    
    snpret = create_lagged_series(
        "^GSPC", datetime.datetime(2001,1,10), 
        datetime.datetime(2005,12,31), lags=5
    )

    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    ]

    # 对参数进行网格搜索
    model = GridSearchCV(SVC(C=1), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    print("Optimised parameters found on training set:")
    print(model.best_estimator_, "\n")
    
    print("Grid scores calculated on training set:")
    for params, mean_score, scores in model.grid_scores_:
        print("%0.3f for %r" % (mean_score, params))
