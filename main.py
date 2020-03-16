#TODO: 1) a tool to combine the scenarios must be created (input only wind scenarios)
#TODO: 2) a tool to create the whole input file must be created (or not?)
#TODO: 3) check the fc tool

from scripts.optimizationUtils.stochasticProgrammingModel import *
from scripts.optimizationUtils.reportingUtils import *
from scripts.optimizationUtils.plotUtils import *

#--- Define basic I/O data
inputFileName = 'data/input2.xlsx'
outputDir = 'reports/'
reportFileName = outputDir+'Report_1.xlsx'
bidFileName = outputDir+'Bid_1.csv'



daP     = pandas.read_excel(inputFileName, sheet_name='DAP', index_col=0)
wind    = pandas.read_excel(inputFileName, sheet_name='Wind', index_col=0)
rPlus   = pandas.read_excel(inputFileName, sheet_name='rplus', index_col=0)
rMinus  = pandas.read_excel(inputFileName, sheet_name='rminus', index_col=0)
probs   = pandas.read_excel(inputFileName, sheet_name='probs', index_col=0)

a, b = stochasticRisk(daP, wind, rPlus, rMinus, probs, 0.95, 0.9)



displayReport(b)
saveReport(b, reportFileName, bidFileName)

plot_bid(b[1])
plot_profit_distribution(b[2], b[0])
plot_hourly_imbalance_dists(b[3], b[4], b[5])