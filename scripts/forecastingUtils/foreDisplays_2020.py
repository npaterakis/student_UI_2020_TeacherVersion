import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy
from scipy import stats

def plot_fit(predicted, actual):
    plt.plot(predicted, 'red', label = 'predicted')
    plt.plot(actual, 'black', label='actual')
    plt.legend()
    plt.xlabel('sample (10-min)')
    plt.ylabel('normalized wind speed')
    plt.show()

def plot_res_autocor(res):
    stdevRes = res.std()
    meanRes = res.mean()
    print('Residual mean and stdev: ', meanRes, stdevRes )
    autocorrelation_plot(res)
    plt.xlabel('lag [10-min]', fontsize=15)
    plt.ylabel('Autocorrelation', fontsize=15)
    plt.title('Autocorrelation of residuals', fontsize=15)
    plt.show()

def plot_res_hist(res):
    binVres = plt.hist(res, bins=50, density=True, color='gray')[0]
    x = numpy.linspace(0 - 4 * res.std(), 0 + 4 * res.std(), 100)
    plt.plot(x, stats.norm.pdf(x, 0, res.std()), 'red')
    plt.xlabel('residual value', fontsize=15)
    plt.ylabel('probability', fontsize=15)
    plt.show()

def plot_windSpeedScenarios(scenarios, arrayActual):
    for j in range(scenarios.shape[0]):
        plt.plot(scenarios[j, :], color='gray')

    plt.plot(arrayActual, color='red', label='Actual')
    plt.plot(numpy.mean(scenarios, axis=0), color='blue', label='Mean scenario')
    plt.xlabel('sample (10-min)', fontsize=15)
    plt.ylabel('speed @100m (m/s)', fontsize=15)
    plt.legend()

    plt.show()

def plot_windPowerScenarios(scenarios, arrayActual):
    for j in range(scenarios.shape[0]):
        plt.plot(scenarios[j, :], color='gray')

    plt.plot(arrayActual[0,:], color='red', label = 'Actual')
    plt.plot(numpy.mean(scenarios, axis=0), color='blue', label='Mean scenario')
    plt.plot(numpy.median(scenarios,axis=0), color='green', label='Median scenario')
    plt.xlabel('sample (hour)', fontsize=15)
    plt.ylabel('Energy (kWh)', fontsize=15)
    plt.legend()

    plt.show()