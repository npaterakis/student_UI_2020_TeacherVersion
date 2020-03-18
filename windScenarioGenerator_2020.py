import numpy
import pandas
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from scripts.forecastingUtils.foreUtils_2020 import *
from scripts.forecastingUtils.foreDisplays_2020 import *
numpy.random.seed(10)

# ---------------------------------------------------------------------------------------------
# -- Basic settings
# ---------------------------------------------------------------------------------------------
inputFileName = 'windSpeed_2020.csv' #Change this to the most updated file
turbineRatedPower = 2500 # in kW
windfarmRatedPower = 25000 #in kW
periodsFuture, periodsPast, daysHistory, numScenarios = 144, (144*3), 30, 30
firstDateTest = '2020-02-10 00:00:00' #Change this to the date for which you need the forecast
firstDateTrain = pandas.to_datetime(firstDateTest)-pandas.Timedelta(str(daysHistory)+'D')

featureSelection = True
plotResidualDiagnostics = False
scenarios = numpy.zeros((numScenarios, periodsFuture)) #initialize matrix of scenarios

#Input/output paths -- do not change
inputDataDir = 'data/'
outputDataDir = 'data/'
outputDataDir1 = firstDateTest.split(' ')[0]
outputFilename = 'wind_'+outputDataDir1+'.csv'

# ---------------------------------------------------------------------------------------------
# -- Data preparation
# ---------------------------------------------------------------------------------------------
# Load and scale data
wspeed = pandas.read_csv(inputDataDir+inputFileName, index_col=0, parse_dates=True, dayfirst=True)['speed']

scaler = StandardScaler()
wspeed.loc[:] = scaler.fit_transform(wspeed.values.reshape(-1,1))[:,0]

# Create datasets
C = createDataSet(wspeed, periodsPast)
trainSet, testSet = splitTrainTest(C, firstDateTrain=firstDateTrain, firstDateTest=firstDateTest, value=10, unit='min')
trainX, trainY = splitXY(trainSet)
testX, testY = splitXY(testSet)
print('Train X: ', trainX.shape, 'Train Y: ', trainY.shape,'Test X: ', testX.shape,'Test Y: ', testY.shape)

# Feature selection
if featureSelection:
    print('Starting feature selection!')
    mask = feature_selection(trainX, trainY, 'rfe')
    trainX = trainX[:,mask]
    print('Done feature selection! New feature matrix size: ', trainX.shape)

else:
    mask = None
    print('No feature selection is applied!')

# ---------------------------------------------------------------------------------------------
# -- Prediction model
# ---------------------------------------------------------------------------------------------
# Generate prediction model
model, res, stdevRes = createPredictionModel(trainX, trainY, method='LR')
print('Residual mean: ', numpy.mean(res), 'Residual stdev: ', stdevRes)

# Plot diagnostics on residuals
if plotResidualDiagnostics:
    plot_fit(model.predict(trainX), trainY)
    plot_res_autocor(res)
    plot_res_hist(res)
else:
    print('No residual diagnostics are plotted!')

# Generate scenarios
for j in range(numScenarios):
    arrayActual, arrayPredicted = forecastForward(testSet, testX, model, scaler, periodsFuture, stdevRes, mask=mask, testY=testY, positivityRequirement=True)

    arrayActual, arrayPredicted = scaler.inverse_transform(arrayActual), scaler.inverse_transform(arrayPredicted)
    scenarios[j, :] = arrayPredicted

plot_windSpeedScenarios(scenarios, arrayActual)
scenariosPower = powerG126(scenarios)
actualPower = powerG126(arrayActual.reshape(1,-1))
plot_windPowerScenarios(scenariosPower, actualPower)

from scripts.generalUtils_2020 import *
outputDir = createDataDirectory(outputDataDir, outputDataDir1)
scenariosPowerDf = pandas.DataFrame(data=(scenariosPower*(windfarmRatedPower/turbineRatedPower))/1000, index=['s'+str(s) for s in range(1, numScenarios+1)], columns=['t'+str(t) for t in range(1, 25)])
scenariosPowerDf.to_csv(outputDir+outputFilename)
generateScenarioTree(outputDataDir, outputDataDir1)
