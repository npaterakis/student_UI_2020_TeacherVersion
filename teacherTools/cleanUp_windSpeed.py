import pandas, numpy
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def powerG126(speed):
    power = pandas.Series(index=speed.index, data=None, dtype='float64')
    speed = pandas.Series(index=speed.index, data=speed['speed'].values)

    for t in speed.index:

        if speed.loc[t] < 2 or speed.loc[t]>25:
            power.loc[t] = 0
        elif speed.loc[t] >=2 and speed.loc[t] < 10:
            power.loc[t] = -7.1754 * (speed.loc[t] ** 3) + 120.13* (speed.loc[t] ** 2)- 252.4* speed.loc[t]+ 186.36
        elif speed.loc[t] >=10 and speed.loc[t] <=21:
            power.loc[t] = 2500
        elif speed.loc[t] >21 and speed.loc[t] <=25:
            power.loc[t] = 9.3333 * (speed.loc[t] ** 3) - 654.31 * (speed.loc[t] ** 2) + 15059 * speed.loc[t] -111619
        else:
            print(t, speed.loc[t])
            raise ValueError

    return power

path = 'C:/Users/npaterakis/Desktop/Data_2020/Wind_ElPerdon/'

A = pandas.read_csv(path+'wind_2018.csv', index_col=0, parse_dates=True, dayfirst=True)
B=  pandas.read_csv(path+'wind_2019.csv', index_col=0, parse_dates=True, dayfirst=True)
C=  pandas.read_csv(path+'wind_2020.csv', index_col=0, parse_dates=True, dayfirst=True)

windSpeed = pandas.concat([A,B,C])*1.21
windPower = powerG126(windSpeed).resample('1H').mean()

windSpeed.to_csv('data/windSpeed_2020.csv')
#windPower.to_csv('windPower_2020.csv')
