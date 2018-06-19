#!/usr/bin/python
# -*- coding: utf-8 -*-

# forecast.py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import sklearn

from pandas_datareader import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC


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
    # 生成S&P500的带有滞后数据的DataFrame
    snpret = create_lagged_series(
    	"^GSPC", datetime.datetime(2001,1,10), 
    	datetime.datetime(2005,12,31), lags=5
    )

    #使用两天以前的数据作为预测基础
    # 使用Direction 作为验证集
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    #数据被分为两部分 一部分用于训练一部分用于验证
    start_test = datetime.datetime(2005,1,1)

    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]
   
    # 批量训练
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()), 
              ("LDA", LDA()), 
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
              	C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
              )]

    # 对每个模型进行训练
    for m in models:
        
        m[1].fit(X_train, y_train)

        # 对每个模型在验证集上进行测试
        pred = m[1].predict(X_test)

        # 打印结果
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(pred, y_test))