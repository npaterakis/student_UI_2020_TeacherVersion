from scripts.optimizationUtils.stochasticProgrammingModel import *
from scripts.optimizationUtils.reportingUtils import *
from scripts.optimizationUtils.plotUtils import *

firstDateTest = '2020-02-10 00:00:00'  #TODO, in the jupyter it will be integrated
folderName = firstDateTest.split(' ')[0]

#--- Define basic I/O data
fileDAP = 'data/'+str(folderName)+'/DAP_'+str(folderName)+'.csv'
fileImNeg = 'data/'+str(folderName)+'/tree_imNeg_'+str(folderName)+'.csv'
fileImPos = 'data/'+str(folderName)+'/tree_imPos_'+str(folderName)+'.csv'
fileWind = 'data/'+str(folderName)+'/tree_wind_'+str(folderName)+'.csv'
fileProbs = 'data/'+str(folderName)+'/tree_probs_'+str(folderName)+'.csv'

outDir = 'data/'+str(folderName)+'/'
reportFileName = outDir+'report_'+str(folderName)+'.xlsx'
bidFileName = outDir+'bid_'+str(folderName)+'.csv'

#--- Load data
daP = pandas.read_csv(fileDAP, index_col=0)
wind = pandas.read_csv(fileWind, index_col=0)
rPlus = pandas.read_csv(fileImPos, index_col=0)
rMinus = pandas.read_csv(fileImNeg, index_col=0)
probs = pandas.read_csv(fileProbs, index_col=0)

#Execute optimization model
a, b = stochasticRisk(daP, wind, rPlus, rMinus, probs, 0.95, 0.1)

displayReport(b)
saveReport(b, reportFileName, bidFileName)

plot_bid(b[1])
plot_profit_distribution(b[2], b[0])
plot_hourly_imbalance_dists(b[3], b[4], b[5])