import pandas, numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def powerG126(speed):
    power = numpy.zeros(speed.shape)
    powerH = numpy.zeros((speed.shape[0], int(speed.shape[1]/6)))

    for s in range(power.shape[0]):
        for t in range(power.shape[1]):
            if speed[s,t] < 2 or speed[s,t]>25:
                power[s,t] = 0
            elif speed[s,t] >=2 and speed[s,t] < 10:
                power[s,t] = -7.1754 * (speed[s,t] ** 3) + 120.13* (speed[s,t] ** 2)- 252.4* speed[s,t]+ 186.36
            elif speed[s,t] >=10 and speed[s,t] <=21:
                power[s,t] = 2500
            elif speed[s,t] >21 and speed[s,t] <=25:
                power[s, t] = 9.3333 * (speed[s, t] ** 3) - 654.31 * (speed[s, t] ** 2) + 15059 * speed[s, t] -111619
            else:
                print(t, s, speed[s,t])
                raise ValueError

    powerDf = pandas.DataFrame(index=['s'+str(s) for s in range(1,power.shape[0]+1)], columns=pandas.date_range('2015-01-01 00:00', periods=power.shape[1], freq='10min'), data=power)
    powerH = powerDf.T.resample('H').mean().T

    #print(powerH)
    return powerH.values

def createDataSet(dfIn, periodsPast):
    dfOut = pandas.DataFrame()

    for t in range(0, periodsPast+1):
        if t == 0:
            colName = 't'
        else:
            colName = 't_'+str(t)

        dfOut[colName] = dfIn.shift(t)

    dfOut = discardNaN(dfOut)

    print('Dataset created!')
    return dfOut[dfOut.columns[::-1]]

def discardNaN(df):
    original_first_period = df.index[0]
    df.dropna(axis=0, inplace= True)
    droped_first_period = df.index[0]
    print('Original: ', original_first_period, 'Remained: ', droped_first_period, '(discarding rows with NaN values)')

    return df

def splitTrainTest(df, firstDateTrain, firstDateTest, value, unit):

    dfTrain = df.loc[firstDateTrain:pandas.to_datetime(firstDateTest)-pandas.Timedelta(value=value, unit=unit)]
    dfTest = df.loc[firstDateTest:]
    print('Dataset was split in train and test set!')
    return dfTrain, dfTest

def splitXY(df):

    X = df.values[:, :-1]
    Y = df.values[:, -1].reshape(-1,1)

    return X, Y

def feature_selection(trainX, trainY, method='rfe'):

    if method == 'mutual_info':
        mi = mutual_info_regression(trainX, trainY, n_neighbors = 10, random_state=1)
        mi = mi/max(mi)
        mask = mi >= 0.2

    if method == 'forest':
        importances = RandomForestRegressor(n_estimators=10, n_jobs=-1).fit(trainX,trainY).feature_importances_
        mask = importances.argsort()[-144:][::-1]

    if method == 'rfe':
        #estimator = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        estimator = LinearRegression(fit_intercept=True)
        selector = RFE(estimator, 6*5, 50, verbose=2).fit(trainX, trainY)
        mask = selector.support_


    return mask

def createPredictionModel(trainX, trainY, method = 'LR'):
    print('Creating base prediction model')

    if method == 'LR':
        model = LinearRegression(fit_intercept=True)

    elif method == 'Ridge':
        model = Ridge(5)

    elif method == 'keras':
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        model = Sequential()
        model.add(Dense(units=3, activation='relu', input_dim=trainX.shape[1]))
        model.add(Dropout(0.01))
        model.add(Dense(units=1))
        model.compile(loss='mse', optimizer='adam')


    else:
        print('Unknown model! Implement yourself?')
        raise ValueError()

    model.fit(trainX, trainY)#, epochs=50, verbose=2) #do not use model = when using keras

    res = model.predict(trainX) - trainY  # calculate residuals
    stdevRes = res.std()  # calculate the standard deviation of the residuals

    return model, res, stdevRes

def forecastForward(testSet, testX, model, scaler, periodsFuture, stdev, mask = None, testY = None, positivityRequirement=True):
    #if positivityRequirement:
    #    print('Positivity of the outcome is enforced!')
    #else:
    #    print('Positivity of the outcome is NOT enforced!')

    list_actual, list_predicted = [], []

    for t_in, t in enumerate(testSet.index):

        if type(testY) != type(None):
            y_actual = testY[t_in]

        if t_in == 0:
            x = testX[t_in, :].reshape((1, -1))
            if type(mask)!= type(None):
                x = x[:,mask]

        else:
            x = testX[t_in, :].reshape((1, -1))
            for el_in, el in enumerate(list_predicted):
                x[:, -len(list_predicted) + el_in] = el

            if type(mask) != type(None):
                x = x[:,mask]

        if positivityRequirement == True:
            basePrediction = model.predict(x)

            y_hat = basePrediction + numpy.random.normal(0, stdev, 1)
            unscaledYhat = scaler.inverse_transform(y_hat)

            while unscaledYhat < 0:
                y_hat = basePrediction + numpy.random.normal(0, stdev, 1)
                unscaledYhat = scaler.inverse_transform(y_hat)

        else:
            y_hat = model.predict(x) + numpy.random.normal(0, stdev, 1)

        list_actual.append(float(y_actual))
        list_predicted.append(float(y_hat))

        if t_in == periodsFuture - 1:
            break

    return numpy.asanyarray(list_actual), numpy.asanyarray(list_predicted)