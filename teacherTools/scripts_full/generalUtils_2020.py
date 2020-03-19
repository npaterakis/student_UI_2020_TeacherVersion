import os, shutil, pandas

def createDataDirectory(topDir, dirName):
    dir = topDir + dirName

    if not os.path.exists(dir):
        print('Creating directory: ', dir)
        os.mkdir(dir)

    else:
        print('Directory already exists! Refreshing its contents: ', dir)
        shutil.rmtree(dir)
        os.mkdir(dir)

    return dir+'/'

def generateScenarioTree(topDir, dirName):
    dir = topDir + dirName

    #imbalance prices
    im = pandas.read_csv(topDir+'ratioScenarios_2020.csv')
    r = im['r']
    rExpanded = pandas.DataFrame(data = None, index = [s for s in range(len(im.index))], columns=['t'+str(t) for t in range(1,25)])
    for col in rExpanded.columns:
        rExpanded.loc[:, col] = r.loc[:].values

    imPos, imNeg = rExpanded.copy(), rExpanded.copy()
    imPos.iloc[rExpanded.loc[:,:] > 1] = 1
    imNeg.iloc[rExpanded.loc[:,:] < 1] = 1

    imbalance_prob = im['prob']

    #Wind
    wind = pandas.read_csv(dir+'/wind_'+dirName+ '.csv')
    # If wind scenarios are not equiprobable, replace the following line to load the appropriate file
    wind_prob = pandas.Series(data = [1/len(wind.index) for s in range(len(wind.index))], index = wind.index)

    numScen = len(im.index) * len(wind.index)

    windNew = pandas.DataFrame(data=None, index=['s'+str(s) for s in range(1,numScen+1)], columns=['t'+str(t) for t in range(1,25)])
    imPosNew = pandas.DataFrame(data=None, index=['s'+str(s) for s in range(1,numScen+1)], columns=['t'+str(t) for t in range(1,25)])
    imNegNew = pandas.DataFrame(data=None, index=['s' + str(s) for s in range(1, numScen + 1)], columns=['t' + str(t) for t in range(1, 25)])
    probNew = pandas.DataFrame(data=None, index=['s' + str(s) for s in range(1, numScen + 1)], columns=['prob'])

    k = 0
    for i in range(len(wind.index)):
        for j in range(len(im.index)):
            #print(k, i,j)
            windNew.iloc[k, :] = wind.iloc[i,:]
            imPosNew.iloc[k, :] = imPos.iloc[j,:]
            imNegNew.iloc[k, :] = imNeg.iloc[j,:]
            probNew.loc['s'+str(k+1),'prob'] = (imbalance_prob.iloc[j] * wind_prob.iloc[i])
            k = k + 1

    windNew.to_csv(dir+'/tree_wind_'+dirName+'.csv')
    imPosNew.to_csv(dir+'/tree_imPos_'+dirName+'.csv')
    imNegNew.to_csv(dir+'/tree_imNeg_'+dirName+'.csv')
    probNew.to_csv(dir+'/tree_probs_'+dirName+'.csv')