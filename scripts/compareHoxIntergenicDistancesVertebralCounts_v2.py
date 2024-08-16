# general script to compare Hox intergenic distances with vertebral counts,
# including PIC
# v2

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath +'hoxData/'
inputPath2 = basePath + 'intergenic/'
outputPath = basePath

#%% import libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import pandas as pd
import scipy.stats
import glob
import scipy.cluster.hierarchy as spc
from skbio import TreeNode
from io import StringIO
import copy
from skbio import TreeNode
from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
colorWheel = [
    "#006BB2",
    "firebrick",
    "orange",
    "green",
    "dodgerblue",
    "maroon",
    "gold",
    "lightseagreen",
    "lightcoral",
    "khaki",
]
colorMap = 'PRGn' # or 'gray' or 'bwr' or 'RdBu'
colorMapVariance = 'Reds' #
import matplotlib.gridspec as gridspec

markerWheel = ['o','s','^','>','<','v','d'] # pour convenience
fontSize = 16
faceColor = 'aliceblue'
markerSize = 2
lineWidth = 0.5
fontToUse = 'Arial'
params = {
   'axes.labelsize': fontSize,
#   'font.family': 'sans-serif',
  # 'font.serif': 'Times New Roman',
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
#   'legend.fontsize': 11,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
#   'figure.figsize': [4.5, 4.5]
   }
mpl.rcParams.update(params)

statistic = 'pearson' # or 'spearman'

removeSnake = True

classKey = 'Class'
speciesKey = 'Species'
nameKey = 'Common Name'

#%% PIC algorithm

def picFunction(tree,x,tipNamesOrig,cols,speciesKey,nameKey,classKey):
    
    # first reduce the tree and x to the species in x which have non-nan data for given columns
    tree,x,tipNamesOrig = reduceTree(tree,x,cols,speciesKey)
    
    if np.size(tipNamesOrig) == 0:
        return np.nan,np.nan
    
    else:
        
        # indColumns = list([0,1,2]) # we want to include the name, species, and class columns
        indColumns = []
        for i in range(len(cols)):
            indColumns = np.append(indColumns,np.array(list(x.columns).index(cols[i])))
        indColumns = np.array(indColumns).astype(int)
        
        xKeys = np.array(list(x.keys()))[indColumns]
        xPICdata = np.zeros((len(tipNamesOrig)-1,len(xKeys)))
        xPICdatan = np.zeros((len(tipNamesOrig)-1,len(xKeys)))
        namePIC = []
        speciesPIC = []
        classPIC = []
        xBank = x.iloc[:,indColumns].to_numpy()
        nameBank = x[nameKey].to_list()
        speciesBank = x[speciesKey].to_list()
        classBank = x[classKey].to_list()
        treeTemp = copy.copy(tree)

        # replace any "nan" in nameBank with "unknown"
        for i in range(len(nameBank)):
            if nameBank[i] != nameBank[i]:
                print(speciesBank[i])
                nameBank[i] = 'unknown'

        # do PIC by looking at nested parentheses in newark tree

        for i in range(len(tipNamesOrig)-1):
            print(i)
            
            # get a string version of the tree
            treeStr = str(treeTemp)
            
            # get tree tip names
            tipNames = []
            for node in treeTemp.tips():
                tipNames.append(node.name)
                
            # find minimally nested parentheses
            found = 0
            indP = treeStr.find('(')
            while found == 0:
                if treeStr[indP+1:].find('(') == -1: # this is the last '('
                    indP2 = indP+treeStr[indP+1:].find(')')+1
                    found = 1
                else:
                    indPb = indP+treeStr[indP+1:].find('(')+1
                    indP2 = indP+treeStr[indP+1:].find(')')+1
                    if indP2 > indPb:
                        indP = indPb
                    else:
                        found = 1
            # just taking inside the parentheses
            tempStr = treeStr[indP+1:indP2]
            tempStr2 = tempStr[tempStr.find(',')+1:]
            picSpecies = ([tempStr[:tempStr.find(':')].replace('_',' '),tempStr2[:tempStr2.find(':')].replace('_',' ')])
            newSpecies = picSpecies[0].replace(' ','_')+'+'+picSpecies[1].replace(' ','_')
            # picNames = ([xPICdf[nameKey][xPICdf[speciesKey]==picSpecies[0]].iloc[0],xPICdf[nameKey][xPICdf[speciesKey]==picSpecies[1]].iloc[0]])
            picNames = ([nameBank[speciesBank.index(picSpecies[0])],nameBank[speciesBank.index(picSpecies[1])]])
            treeTimes = []
            treeTimes.append(float(tempStr[len(picSpecies[0])+1:tempStr.find(',')]))
            treeTimes.append(float(tempStr2[len(picSpecies[1])+1:]))
            
            # get the classes
            # picClasses = ([xPICdf[classKey][xPICdf[speciesKey]==picSpecies[0]].iloc[0],xPICdf[classKey][xPICdf[speciesKey]==picSpecies[1]].iloc[0]])
            picClasses = ([classBank[speciesBank.index(picSpecies[0])],classBank[speciesBank.index(picSpecies[1])]])
            
            indP2b = indP2+treeStr[indP2:].find(':')
            
            # replace the entire parentheses + the node name to the new name
            treeStr = treeStr[:indP]+newSpecies+treeStr[indP2b:]
            
            # new way
            xleft = xBank[speciesBank.index(picSpecies[0]),:]
            xright = xBank[speciesBank.index(picSpecies[1]),:]

            xPICdata[i,:] = xleft - xright
            if (treeTimes[0]==0) | (treeTimes[1]==0): # just take even average
                xPICdatan[i,:] = (xleft - xright)/np.sqrt(1+1)
                xBank = np.append(xBank,(xleft[None,:]+xright[None,:])/2,axis=0)
            else:
                xPICdatan[i,:] = (xleft - xright)/np.sqrt(treeTimes[0]+treeTimes[1])
                xBank = np.append(xBank,(treeTimes[1]*xleft[None,:]+treeTimes[0]*xright[None,:])/(treeTimes[0]+treeTimes[1]),axis=0)
            
            nameBank.append(picNames[0]+'+'+picNames[1])
            speciesBank.append(picSpecies[0]+'+'+picSpecies[1])
            classBank.append(picClasses[0]+'+'+picClasses[1])
            namePIC.append('('+picNames[0]+')-('+picNames[1]+')')
            speciesPIC.append('('+picSpecies[0]+')-('+picSpecies[1]+')')
            classPIC.append('('+picClasses[0]+')-('+picClasses[1]+')')
            
            if i < len(tipNamesOrig)-2:
                # adjust the new branch length (Step 4 in Felsenstein 1985 recipe)
                # this is just the branch length just after this "newSpecies", so find that
                indP3 = treeStr.find(newSpecies)+len(newSpecies)+1
                # determine whichever comes first, a comma or a )
                test0 = treeStr[indP3:].find(',')
                if test0 == -1:
                    test0 = 1000000
                test1 = treeStr[indP3:].find(')')
                indP3b = indP3 + min(test0,test1)
                oldBranchLength = float(treeStr[indP3:indP3b])
                if (treeTimes[0]==0) | (treeTimes[1]==0):
                    newBranchLength = oldBranchLength
                else:
                    newBranchLength = oldBranchLength + treeTimes[0]*treeTimes[1]/(treeTimes[0]+treeTimes[1])
                # replace
                treeStr = treeStr[:indP3]+str(newBranchLength)+treeStr[indP3b:]
            
            if i < len(tipNamesOrig)-2:
                # pass the new tree to the next round
                treeNew = TreeNode.read(StringIO(treeStr))
                treeTemp = treeNew
                
        xPIC = pd.DataFrame()
        xPIC[nameKey] = namePIC
        xPIC[speciesKey] = speciesPIC
        xPIC[classKey] = classPIC
        newCols = []
        for i in range(len(xKeys)):
            newCols.append(pd.Series(xPICdata[:,i], name=xKeys[i], index=xPIC.index))
            # xPIC[xKeys[i]] = list(xPICdata[:,i])
        xPIC = pd.concat([xPIC,*newCols],axis=1)
            
        xPICn = pd.DataFrame()
        xPICn[nameKey] = namePIC
        xPICn[speciesKey] = speciesPIC
        xPICn[classKey] = classPIC
        newCols2 = []
        for i in range(len(xKeys)):
            newCols2.append(pd.Series(xPICdatan[:,i], name=xKeys[i], index=xPIC.index))
            # xPIC[xKeys[i]] = list(xPICdata[:,i])
        xPICn = pd.concat([xPICn,*newCols2],axis=1)
        
        return xPICn,xPIC

#%% reduce tree and x to the species in x which have non-nan data for given columns

def reduceTree(tree,x,cols,speciesKey):

    indColumns = list([0,1,2]) # we want to include the name, species, and class columns
    for i in range(len(cols)):
        indColumns = np.append(indColumns,np.array(list(x.columns).index(cols[i])))
    indColumns = np.array(indColumns).astype(int)
    
    x2 = x.iloc[:,indColumns]
    x2 = x2.dropna()
    x2 = x2.reset_index(drop=True)
    
    if len(x2) == 0:
        return np.nan,np.nan,np.nan
    
    else:
        
        # shear the tree
        tree2 = tree.shear(x2[speciesKey].values.tolist())
        tree2.prune()
        
        # get the new tip names
        tipNamesOrig2 = []
        for node in tree2.tips():
            tipNamesOrig2.append(node.name)
        tipNamesOrig2 = np.array(tipNamesOrig2)
        
        return tree2,x2,tipNamesOrig2
    
#%% a function for saving the two traits that should be compared along with their pruned tree

def saveForPIC(speciesList,feature0,feature1,database,treeToSave,suffix):

    # dataframe for saving
    saveData = pd.DataFrame(columns=['Species',feature0,feature1])
    # check if the 'treeSpecies' column is in the database, otherwise make it
    if 'treeSpecies' not in database.columns:
        database['treeSpecies'] = [database['Species'].iloc[i].replace(' ','_') for i in range(len(database))]
    saveData['treeSpecies'] = database['treeSpecies'][database['Species'].isin(speciesList)]
    saveData[feature0] = database[feature0][database['Species'].isin(speciesList)]
    saveData[feature1] = database[feature1][database['Species'].isin(speciesList)]
    # save this data for double-checking the PIC with other software
    saveData.to_csv(outputPath+'intergenic/intergenicData_'+feature0+'_'+feature1+suffix+'.csv',index=False)
    # get a reduced version of the tree corresponding to these species
    treeReduced = treeToSave.copy()
    # prune the tree to only include the species in the vertebralData and save
    treeReduced = treeReduced.shear(speciesList)
    treeReduced.prune()
    treeReduced.write(outputPath+'intergenic/intergenicData_'+feature0+'_'+feature1+suffix+'_tree.nwk',format='newick')

#%% function to get the number of asterisks to use for the p-value

def getAsterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

#%% load the intergenic data

# pre PIC
intergenicAll = pd.read_csv(inputPath+'speciesWithGenomeIntergenicDistances_v2.csv')
if removeSnake:
    speciesToRemove = (['Crotalus tigris',
                        'Thamnophis elegans',
                        'Pantherophis guttatus'])
    for species in speciesToRemove:
        intergenicAll = intergenicAll[intergenicAll[speciesKey] != species]
intergenicAll = intergenicAll.reset_index(drop=True)

# post PIC
if removeSnake:
    suffix = '_noSnakes'
else:
    suffix = ''
vertebralConstraints = np.array(['C','T','L','S','Ca'])
intergenicCervicalPIC = pd.read_csv(inputPath2+'pic_normalized_intergenic_Cervical'+suffix+'.csv')
intergenicThoracicPIC = pd.read_csv(inputPath2+'pic_normalized_intergenic_Thoracic'+suffix+'.csv')
intergenicLumbarPIC = pd.read_csv(inputPath2+'pic_normalized_intergenic_Lumbar'+suffix+'.csv')
intergenicSacralPIC = pd.read_csv(inputPath2+'pic_normalized_intergenic_Sacral'+suffix+'.csv')
intergenicCaudalPIC = pd.read_csv(inputPath2+'pic_normalized_intergenic_Caudal'+suffix+'.csv')

#%% rename the Cervical->C, Thoracic->T, Lumbar->L, Sacral->S, Caudal->Ca
# for all these intergenic dataframes
intergenicAll = intergenicAll.rename(columns={'Cervical':'C','Thoracic':'T','Lumbar':'L','Sacral':'S','Caudal':'Ca'})
intergenicCervicalPIC = intergenicCervicalPIC.rename(columns={'Cervical':'C'})
intergenicThoracicPIC = intergenicThoracicPIC.rename(columns={'Thoracic':'T'})
intergenicLumbarPIC = intergenicLumbarPIC.rename(columns={'Lumbar':'L'})
intergenicSacralPIC = intergenicSacralPIC.rename(columns={'Sacral':'S'})
intergenicCaudalPIC = intergenicCaudalPIC.rename(columns={'Caudal':'Ca'})

#%% separate the vertebral data into classes

mammals = intergenicAll[intergenicAll[classKey]=='Mammalia']
birds = intergenicAll[intergenicAll[classKey]=='Aves']
reptiles = intergenicAll[intergenicAll[classKey]=='Reptilia']
amphibians = intergenicAll[intergenicAll[classKey]=='Amphibia']

#%% get the tree for the PIC here

treeFileName = inputPath+'hoxIntergenicTree.nwk'

with open(treeFileName) as f:
    treeFile = f.read()
tree = TreeNode.read(StringIO(treeFile))

# species in tree
tipNamesOrig = []
for node in tree.tips():
    tipNamesOrig.append(node.name)


#%% we should loop through the vertebral plasticities and compare the intergenic pre- and post-PICs
# since the database of the vertebral and intergenic is not the same,
# I'm going to compare at the Class level
# so if there are any for this plasticity, I will extract the classes from the intergenic data
# and then compare the vertebral data to the intergenic data for each class
# and for all combined

plasticities = pd.read_csv(outputPath+'vertebral/plasticityMasterUnique_organized.csv')
vertebralCols = (['C','T','L','S','Ca'])
intergenicAllKeys = list(intergenicAll.keys())
indCols = np.arange(intergenicAllKeys.index('A1'),intergenicAllKeys.index('D13')+1)
intergenicCols = np.array(intergenicAllKeys)[indCols]
speciesList = []
corrMat = np.zeros((len(plasticities),len(intergenicCols)))
pMat = np.zeros((len(plasticities),len(intergenicCols)))
corrMatPIC = np.zeros((len(plasticities),len(intergenicCols)))
pMatPIC = np.zeros((len(plasticities),len(intergenicCols)))
plasticityFormulaList = []
plasticityLabelList = []
dataColList = []
classesInvolvedList = []

for i in range(len(plasticities)):
    print(i)
    # find nonzero C,T,L,S,Ca columns in plastiticies.iloc[i]
    # these are the datacols of interest
    dataCols = []
    plasticityFormula = np.zeros(5)
    plasticityLabel = ''
    if plasticities['C'].iloc[i] != 0:
        dataCols.append('C')
        plasticityFormula[0] = plasticities['C'].iloc[i]
        if plasticities['C'].iloc[i] < 0:
            plusMinus = '-'
        else:
            plusMinus = '+'
        plasticityLabel = plasticityLabel + plusMinus + str(int(plasticities['C'].iloc[i])) + 'C'
    if plasticities['T'].iloc[i] != 0:
        dataCols.append('T')
        plasticityFormula[1] = plasticities['T'].iloc[i]
        if plasticities['T'].iloc[i] < 0:
            plusMinus = '-'
        else:
            plusMinus = '+'
        plasticityLabel = plasticityLabel + plusMinus + str(int(plasticities['T'].iloc[i])) + 'T'
    if plasticities['L'].iloc[i] != 0:
        dataCols.append('L')
        plasticityFormula[2] = plasticities['L'].iloc[i]
        if plasticities['L'].iloc[i] < 0:
            plusMinus = '-'
        else:
            plusMinus = '+'
        plasticityLabel = plasticityLabel + plusMinus + str(int(plasticities['L'].iloc[i])) + 'L'
    if plasticities['S'].iloc[i] != 0:
        dataCols.append('S')
        plasticityFormula[3] = plasticities['S'].iloc[i]
        if plasticities['S'].iloc[i] < 0:
            plusMinus = '-'
        else:
            plusMinus = '+'
        plasticityLabel = plasticityLabel + plusMinus + str(int(plasticities['S'].iloc[i])) + 'S'
    if plasticities['Ca'].iloc[i] != 0:
        dataCols.append('Ca')
        plasticityFormula[4] = plasticities['Ca'].iloc[i]
        if plasticities['Ca'].iloc[i] < 0:
            plusMinus = '-'
        else:
            plusMinus = '+'
        plasticityLabel = plasticityLabel + plusMinus + str(int(plasticities['Ca'].iloc[i])) + 'Ca'
    plasticityFormulaList.append(plasticityFormula)
    plasticityLabel = plasticityLabel[1:]
    # remove any '1s'
    plasticityLabel = plasticityLabel.replace('1','')
    plasticityLabel = plasticityLabel.replace('--','-')
    plasticityLabelList.append(plasticityLabel)
    dataColList.append(dataCols)
    
    # get the yTest data
    classesInvolvedListTemp = []
    speciesTest = []
    if plasticities['numMammals'].iloc[i] > 0:
        mammalsTest = intergenicAll[intergenicAll[classKey]=='Mammalia']
        # reduce to those which have no nans in the dataCols
        mammalsTest = mammalsTest.dropna(subset=dataCols)
        for j in range(len(mammalsTest)):
            speciesTest.append(mammalsTest[speciesKey].iloc[j])
        classesInvolvedListTemp.append('Mammalia')
    else:
        mammalsTest = 0
    if plasticities['numBirds'].iloc[i] > 0:
        birdsTest = intergenicAll[intergenicAll[classKey]=='Aves']
        # reduce to those which have no nans in the dataCols
        birdsTest = birdsTest.dropna(subset=dataCols)
        for j in range(len(birdsTest)):
            speciesTest.append(birdsTest[speciesKey].iloc[j])
        classesInvolvedListTemp.append('Aves')
    else:
        birdsTest = 0
    if plasticities['numReptiles'].iloc[i] > 0:
        reptilesTest = intergenicAll[intergenicAll[classKey]=='Reptilia']
        # reduce to those which have no nans in the dataCols
        reptilesTest = reptilesTest.dropna(subset=dataCols)
        for j in range(len(reptilesTest)):
            speciesTest.append(reptilesTest[speciesKey].iloc[j])
        classesInvolvedListTemp.append('Reptilia')
    else:
        reptilesTest = 0
    if plasticities['numAmphibians'].iloc[i] > 0:
        amphibiansTest = intergenicAll[intergenicAll[classKey]=='Amphibia']
        # reduce to those which have no nans in the dataCols
        amphibiansTest = amphibiansTest.dropna(subset=dataCols)
        for j in range(len(amphibiansTest)):
            speciesTest.append(amphibiansTest[speciesKey].iloc[j])
        classesInvolvedListTemp.append('Amphibia')
    else:
        amphibiansTest = 0
    speciesList.append(speciesTest) # save for later
    allTest = intergenicAll[intergenicAll[speciesKey].isin(speciesTest)]
    yTest = np.dot(plasticityFormula[plasticityFormula!=0],allTest[dataCols].to_numpy().T)
    classesInvolvedList.append(classesInvolvedListTemp)
        
    # loop through the intergenic data and get the pre- and post-PICs
    # then determine the correlation between the vertebral and intergenic pre- and post-PICs
    
    for j in range(len(intergenicCols)):
        
        # if all the data is nan, then skip
        if len(allTest[intergenicCols[j]][~np.isnan(allTest[intergenicCols[j]])]) == 0:
            corrMat[i,j] = np.nan
            pMat[i,j] = np.nan
            corrMatPIC[i,j] = np.nan
            pMatPIC[i,j] = np.nan
            continue
        
        print(intergenicCols[j])
        # get the intergenic data
        
        xTest = allTest[intergenicCols[j]]
        if len(xTest[~np.isnan(xTest)]) > 3:
            r,p = scipy.stats.pearsonr(xTest[(~np.isnan(xTest))],yTest[(~np.isnan(xTest))])
        else:
            r = np.nan
            p = np.nan
        # do the PIC
        dataColsTemp = np.append(dataCols,intergenicCols[j])
        xPICn,xPIC = picFunction(tree,allTest,speciesTest,dataColsTemp,speciesKey,nameKey,classKey)
        if np.size(xPICn) == 1:
            rPIC = np.nan
            pPIC = np.nan
        else:
            yTestPIC = np.dot(plasticityFormula[plasticityFormula!=0],xPICn[dataCols].to_numpy().T)
            xTestPIC = xPICn[intergenicCols[j]]
            if len(xTestPIC[~np.isnan(xTestPIC)]) > 3:
                rPIC,pPIC = scipy.stats.pearsonr(xTestPIC[~np.isnan(xTestPIC)],yTestPIC[~np.isnan(xTestPIC)])
            else:
                rPIC = np.nan
                pPIC = np.nan
        # save to the matrices
        corrMat[i,j] = r
        pMat[i,j] = p
        corrMatPIC[i,j] = rPIC
        pMatPIC[i,j] = pPIC
        
        
#%% just go through each vertebral column individually and plot the correlation matrix
# between this and all the intergenic distances (just prePIC and for all classes)

corrMatAll = np.zeros((len(vertebralCols),len(intergenicCols)))
for i in range(len(vertebralCols)):
    xTest = intergenicAll[vertebralCols[i]]
    for j in range(len(intergenicCols)):
        yTest = intergenicAll[intergenicCols[j]]
        if len(xTest[(~np.isnan(xTest))&(~np.isnan(yTest))]) > 3:
            r,p = scipy.stats.pearsonr(xTest[(~np.isnan(xTest))&(~np.isnan(yTest))],yTest[(~np.isnan(xTest))&(~np.isnan(yTest))])
        else:
            r = np.nan
            p = np.nan
        corrMatAll[i,j] = r
        
# do the same thing for all 4 classes

corrMatMammals = np.zeros((len(vertebralCols),len(intergenicCols)))
corrMatBirds = np.zeros((len(vertebralCols),len(intergenicCols)))
corrMatReptiles = np.zeros((len(vertebralCols),len(intergenicCols)))
corrMatAmphibians = np.zeros((len(vertebralCols),len(intergenicCols)))
for i in range(len(vertebralCols)):
    xTestMammals = mammals[vertebralCols[i]]
    xTestBirds = birds[vertebralCols[i]]
    xTestReptiles = reptiles[vertebralCols[i]]
    xTestAmphibians = amphibians[vertebralCols[i]]
    for j in range(len(intergenicCols)):
        yTestMammals = mammals[intergenicCols[j]]
        yTestBirds = birds[intergenicCols[j]]
        yTestReptiles = reptiles[intergenicCols[j]]
        yTestAmphibians = amphibians[intergenicCols[j]]
        if len(xTestMammals[(~np.isnan(xTestMammals))&(~np.isnan(yTestMammals))]) > 3:
            rMammals,pMammals = scipy.stats.pearsonr(xTestMammals[(~np.isnan(xTestMammals))&(~np.isnan(yTestMammals))],yTestMammals[(~np.isnan(xTestMammals))&(~np.isnan(yTestMammals))])
        else:
            rMammals = np.nan
            pMammals = np.nan
        if len(xTestBirds[(~np.isnan(xTestBirds))&(~np.isnan(yTestBirds))]) > 3:
            rBirds,pBirds = scipy.stats.pearsonr(xTestBirds[(~np.isnan(xTestBirds))&(~np.isnan(yTestBirds))],yTestBirds[(~np.isnan(xTestBirds))&(~np.isnan(yTestBirds))])
        else:
            rBirds = np.nan
            pBirds = np.nan
        if len(xTestReptiles[(~np.isnan(xTestReptiles))&(~np.isnan(yTestReptiles))]) > 3:
            rReptiles,pReptiles = scipy.stats.pearsonr(xTestReptiles[(~np.isnan(xTestReptiles))&(~np.isnan(yTestReptiles))],yTestReptiles[(~np.isnan(xTestReptiles))&(~np.isnan(yTestReptiles))])
        else:
            rReptiles = np.nan
            pReptiles = np.nan
        if len(xTestAmphibians[(~np.isnan(xTestAmphibians))
                                 &(~np.isnan(yTestAmphibians))]) > 3:
                rAmphibians,pAmphibians = scipy.stats.pearsonr(xTestAmphibians[(~np.isnan(xTestAmphibians))&(~np.isnan(yTestAmphibians))],yTestAmphibians[(~np.isnan(xTestAmphibians))&(~np.isnan(yTestAmphibians))])
        else:
            rAmphibians = np.nan
            pAmphibians = np.nan
        corrMatMammals[i,j] = rMammals
        corrMatBirds[i,j] = rBirds
        corrMatReptiles[i,j] = rReptiles
        corrMatAmphibians[i,j] = rAmphibians

#%% now another another try but squeezing the Hox gene vs. transition plot and fitting in a 
# plot with the correlations between the intergenic distances and vertebral numbers

# reduce the font size
fontSize = 12
markerSize = 5
# set these using mpl
plt.rcParams.update({'font.size': fontSize})

scaleRGB = 256
colorWheelRGB = [
    ([0,107,178]),
    ([178,34,34]),
    ([255,165,0]),
    ([0,128,0])
]
hoxLetters = (['A','B','C','D'])
hoxGenes = np.arange(1,13+1,1).astype(int)
# imageWidth = 0.2
# imageHeight = 0.25
imageHeight = 0.90/len(hoxLetters)
# colorMap = 'coolwarm'

# load data

correlation = pd.read_csv(inputPath+'correlationStudies_v2.csv')
# get unique list of organisms
organisms = list(correlation['Organism'].unique())
transitions = (['C.T', 'T.L', 'T.S', 'L.S', 'S.Ca']) # doing this manually for now...
vertebrae = (['C','T','L','S','Ca'])

# load images

images = []
imageMouse = plt.imread(inputPath+'speciesImages/musMusculus.png')
images.append(imageMouse)
imageChicken = plt.imread(inputPath+'speciesImages/gallusGallus.png')
images.append(imageChicken)
imageAlligator = plt.imread(inputPath+'speciesImages/crocodylus.png')
images.append(imageAlligator)
imageSnake = plt.imread(inputPath+'speciesImages/lampropeltisCaliforniae.png')
images.append(imageSnake)
imageLizard = plt.imread(inputPath+'speciesImages/agamaAgama.png')
images.append(imageLizard)
imageCaecilian = plt.imread(inputPath+'speciesImages/siphonopsAnnulatus.png')
images.append(imageCaecilian)
imageTurtle = plt.imread(inputPath+'speciesImages/graptemysPseudogeographica.png')
images.append(imageTurtle)
imageOstrich = plt.imread(inputPath+'speciesImages/struthioCamelus.png')
images.append(imageOstrich)
imageZebraFinch = plt.imread(inputPath+'speciesImages/taeniopygiaGuttata.png')
images.append(imageZebraFinch)

imageHeights = imageHeight*np.array([1,1.4,1,1.8,2.0,1,1,1.7,1])

# make these into an easy access collection
imageCollection = []
aspectRatio = []
for i in range(len(organisms)):
    aspectRatio.append(images[i].shape[1]/images[i].shape[0])
    temp = []
    for j in range(len(hoxLetters)):
        imageTemp = copy.deepcopy(images[i])
        for k in range(3):
            imageTemp[:,:,k] = (colorWheelRGB[j][k]/scaleRGB)*images[i][:,:,3]
        temp.append(imageTemp)
    imageCollection.append(temp)
    
# plot with gridspec with 2 rows and 2 columns
# the first column will be the hox literature studies (one plot)
# the second column and first and second rows will be the C and T vs. B9 (two plots)

plotAspectRatio = 2.0/8
plotSize = 1
# fig, ax = plt.subplots(figsize=(plotSize,plotSize/plotAspectRatio))

# Create a figure and a gridspec layout
fig = plt.figure(figsize=(12,6))#,constrained_layout=False)
fig.subplots_adjust(hspace=1.0,wspace=3.0)
gs = gridspec.GridSpec(6,10)
ax0a = plt.subplot(gs[:,0:2])
ax0 = plt.subplot(gs[:,2:4])
ax0b = plt.subplot(gs[:,4:6])
ax1 = plt.subplot(gs[:2,6:8])
ax2 = plt.subplot(gs[2:4,6:8])
ax3 = plt.subplot(gs[:2,8:10])
ax4 = plt.subplot(gs[2:4,8:10])

# this plot is arranged by hox gene
geneNum = []
geneCluster = []
for i in range(len(correlation)):
    temp = correlation['Gene'].iloc[i].split('.')
    geneNum.append(int(temp[1]))
    geneCluster.append(temp[0])
correlation['GeneNum'] = np.array(geneNum).astype(int)
correlation['Cluster'] = geneCluster
    
correlationTransitions = correlation[~correlation['Transition'].isnull()]
# we don't care about the "vertebrae" column now, so remove duplicates based on the other columns
correlationTransitions = correlationTransitions.drop_duplicates(subset=['Organism','Gene','Transition'])
# reset index
correlationTransitions = correlationTransitions.reset_index(drop=True)
    
maxHorizontal = 3
maxVertical = 3 # the maximum number is 9 so this should be enough...
xLoc = np.array([0.2,0.525,0.775,0.4,0.4,0.675,0.7,0.1,0.1])-0.45
yLoc = np.array([0.5,0.5,0.5,0.2,0.7,0.2,0.7,0.2,0.7])-0.5
# make a reduced version for 13 since they are spilling over
xLoc13 = np.array([0.2,0.4,0.1,0.1])-0.45
yLoc13 = np.array([0.5,0.7,0.2,0.7])-0.5
for i in range(len(hoxGenes)):
    temp = correlationTransitions[correlationTransitions['GeneNum']==i+1]
    if len(temp) == 0:
        continue
    else:
        for j in range(len(transitions)):
            temp2 = temp[temp['Transition']==transitions[j]]
            if len(temp2) == 0:
                continue
            else:
                cnt = 0
                for k in range(len(temp2)):
                    indOrganism = organisms.index(temp2['Organism'].iloc[k])
                    indTransition = transitions.index(temp2['Transition'].iloc[k])
                    indHoxLetter = hoxLetters.index(temp2['Cluster'].iloc[k])
                    if i < len(hoxGenes)-1:
                        xpos = indTransition+xLoc[cnt]
                        ypos = (i+1)+yLoc[cnt]
                    else:
                        xpos = indTransition+xLoc13[cnt]
                        ypos = (i+1)+yLoc13[cnt]
                    ax0.imshow(imageCollection[indOrganism][indHoxLetter], extent=[xpos-((1/plotAspectRatio)*(5/13)*(imageHeights[indOrganism]*aspectRatio[indOrganism]))/2,xpos+((1/plotAspectRatio)*(5/13)*(imageHeights[indOrganism]*aspectRatio[indOrganism]))/2,ypos-imageHeights[indOrganism]/2,ypos+imageHeights[indOrganism]/2], aspect='auto', zorder=10)
                    cnt = cnt + 1
                
ax0.set_xlabel('vertebral transition',fontsize=fontSize)
xLabels = (['C/T','T/L','T/S','L/S','S/Ca'])
ax0.set_xticks(np.arange(len(transitions)))
ax0.set_xticklabels(xLabels,rotation=45,fontsize=fontSize)
ax0.set_xlim(-0.5,len(transitions)-0.5)
ax0.set_yticks(np.arange(1,13+1))
ax0.set_yticklabels(hoxGenes,fontsize=fontSize)
ax0.set_ylim(0.5,13.5)
ax0.set_title('knock-out/in experiments\nanterior expression',fontsize=fontSize)

from matplotlib.legend_handler import HandlerLine2D
def update_prop(handle, orig):
    handle.update_from(orig)
    handle.set_marker("")
ax0.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)},frameon=True,loc='lower right',facecolor='white',framealpha=1,edgecolor='white')

# lines to guide the eye for the Hox gene divisions
for i in range(1,13):
    ax0.plot(np.linspace(-1,len(transitions)+1,100),(i+0.5)*np.ones(100),'k--',linewidth=0.5)
    
# lines to guide the eye for the transition divisions
for i in range(0,len(transitions)):
    ax0.plot((i+0.5)*np.ones(100),np.linspace(-1,13+1,100),'k--',linewidth=0.5)

ax0.text(len(transitions)-1.25-0.35-0.9,1-0.05,"A", color=colorWheel[0], va='center', ha='left', zorder=10,fontsize=fontSize)
ax0.text(len(transitions)-1.25-0.15-0.6,1-0.05,"B", color=colorWheel[1], va='center', ha='left', zorder=10,fontsize=fontSize)
ax0.text(len(transitions)-1.25+0.05-0.3,1-0.05,"C", color=colorWheel[2], va='center', ha='left', zorder=10,fontsize=fontSize)
ax0.text(len(transitions)-1.25+0.25,1-0.05,"D", color=colorWheel[3], va='center', ha='left', zorder=10,fontsize=fontSize)

# plot the correlation matrix
corrMatAllRearranged = np.zeros((int(len(vertebralCols)*4),int(len(intergenicCols)/4)))
for i in range(len(vertebralCols)):
    corrMatAllRearranged[4*i+0,:] = corrMatAll[i,:int(len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+1,:] = corrMatAll[i,int(len(intergenicCols)/4):int(len(intergenicCols)/2)]
    corrMatAllRearranged[4*i+2,:] = corrMatAll[i,int(len(intergenicCols)/2):int(3*len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+3,:] = corrMatAll[i,int(3*len(intergenicCols)/4):]
# plot the matrix
# put the origin at the bottom left
im = ax0b.imshow(corrMatAllRearranged.T, cmap=colorMap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',origin='lower')

ax0b.set_xticks(np.arange(len(vertebralCols))*4+1.5)
ax0b.set_xticklabels(vertebralCols,rotation=0,y=-0.01,fontsize=fontSize)
ax0b.set_yticks(np.arange(13)-0.0)
ax0b.set_yticklabels(np.arange(13)+1,fontsize=fontSize)
ax0b.set_ylabel('$\it{Hox}$ intergenic distance',fontsize=fontSize)
ax0b.set_xlabel('vertebrae',fontsize=fontSize)

# plot vertical dashed lines to separate the plasticities
for i in range(len(vertebralCols)-1):
    ax0b.plot([4*(i+1)-0.5,4*(i+1)-0.5],[-0.5,12.5],'k--',linewidth=lineWidth)

# put some ticks labels (no ticks) on the top for the Hox clusters A,B,C,D repeating 5 times (one for each vertebral column)
ax0b2 = ax0b.twiny()
ax0b2.set_xlim(ax0b.get_xlim())
ax0b2.set_xticks(np.arange(len(vertebralCols)*4)+0.0)
hoxLetters = (['A','B','C','D'])
hoxLettersLabels = []
for i in range(len(vertebralCols)):
    for j in range(4):
        hoxLettersLabels.append(hoxLetters[j])
# ax0b2.set_xticklabels(hoxLettersLabels,rotation=45,fontsize=fontSize,horizontalalignment='left')
ax0b2.set_xticklabels(['']*len(hoxLettersLabels))
ax0b.text(-.05,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.06,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.16,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.275,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(-.05+0.22,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.06+0.22,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.16+0.22,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.275+0.22,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(-.05+0.44,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.06+0.44,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.16+0.44,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.275+0.44,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(-.05+0.66,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.06+0.66,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.16+0.66,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.275+0.66,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(-.05+0.88,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.06+0.88,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.16+0.88,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0b.text(.275+0.88,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)

# plot arrows to show the HoxB9 to C and T correlation
# rotate it by 45 degrees
ax0b.quiver(4,8.7,-13/5,-5/13, scale_units='xy', angles='xy', scale=1, color='k',width=0.015)
ax0b.quiver(4+4,8.7,-13/5,-5/13, scale_units='xy', angles='xy', scale=1, color='k',width=0.015)
# and now also the HoxB9 to L and S correlations
ax0b.quiver(12,8.7,-13/5,-5/13, scale_units='xy', angles='xy', scale=1, color='k',width=0.015)
ax0b.quiver(12+4,8.7,-13/5,-5/13, scale_units='xy', angles='xy', scale=1, color='k',width=0.015)

# add the colorbar to the left
cbar = fig.colorbar(im, ax=ax0b, orientation='vertical',pad=0.035,shrink=0.3,aspect=30)
# cbar.set_label('Pearson correlation coefficient', rotation=90,fontsize=fontSize)
# put the cbar label on top
cbar.set_label('r', rotation=0,fontsize=fontSize,y=1.15,labelpad=-20)
# set the ticks to be integers -1,0,1
cbar.set_ticks([-1,0,1])
# set the colorbar tick label font size
cbar.ax.tick_params(labelsize=fontSize)

# load the intergenic and vertebral PIC data
intergenicCervicalPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Cervical_B9'+suffix+'.csv')
intergenicThoracicPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Thoracic_B9'+suffix+'.csv')
intergenicLumbarPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Lumbar_B9'+suffix+'.csv')
intergenicSacralPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Sacral_B9'+suffix+'.csv')

# plot the B9 distance and Cervical
markerSize = 6
ax1.plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['C'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax1.plot((intergenicAll['B9'][intergenicAll[classKey]=='Aves'])/1000,intergenicAll['C'][intergenicAll[classKey]=='Aves'],marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax1.plot((intergenicAll['B9'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')])/1000,intergenicAll['C'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax1.set_ylabel('Cervical',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[nameKey]!='snake']
y = intergenicAll['C'][intergenicAll[nameKey]!='snake']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
xPIC = intergenicCervicalPICB9['B9']
yPIC = intergenicCervicalPICB9['Cervical']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
# ax1.set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax1.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax1.set_xlim(-1,150)
ax1.set_xticks((0,50,100,150))
ax1.set_xticklabels([])
ax1.set_ylim(0,25)
# set tick font size
ax1.tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax1.annotate('Mammalia',
    xy=(0.94,0.85), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right',fontsize=fontSize-1)
ax1.annotate('Aves',
    xy=(0.94,0.72), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right',fontsize=fontSize-1)
ax1.annotate('Reptilia',
    xy=(0.94,0.59), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[2],ha='right',fontsize=fontSize-1)
saveForPIC(intergenicAll['Species'].to_list(),'B9','C',intergenicAll,tree,'_full')

# plot the B9 distance and Thoracic
ax2.plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['T'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize)
ax2.plot((intergenicAll['B9'][intergenicAll[classKey]=='Aves'])/1000,intergenicAll['T'][intergenicAll[classKey]=='Aves'],marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize)
ax2.plot((intergenicAll['B9'][intergenicAll[classKey]=='Reptilia'])/1000,intergenicAll['T'][intergenicAll[classKey]=='Reptilia'],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize)
ax2.set_ylabel('Thoracic',fontsize=fontSize)
ax2.set_xlabel('kb',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[nameKey]!='snake']
y = intergenicAll['T'][intergenicAll[nameKey]!='snake']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
xPIC = intergenicThoracicPICB9['B9']
yPIC = intergenicThoracicPICB9['Thoracic']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)

ax2.set_yticks((0,10,20,30))
# ax2.set_xlim(-1,250)
ax2.set_xlim(-1,150)
# ax2.set_xticks((0,100,200))
ax2.set_xticks((0,50,100,150))
ax2.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax2.set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'*',fontsize=fontSize)
ax2.tick_params(axis='both', which='major', labelsize=fontSize)
print('T vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
# ax2.text(0.58,0.65,'tuatara',fontsize=fontSize,fontweight='normal',transform=ax2.transAxes)
saveForPIC(intergenicAll['Species'].to_list(),'B9','T',intergenicAll,tree,'_full')

# plot subplot labels ("a", "b", etc.)
ax0.text(-.4,1.01,'B',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)
ax0b.text(-.4,1.01,'C',fontsize=fontSize+4,fontweight='normal',transform=ax0b.transAxes)
ax1.text(-0.4,1.02,'D',fontsize=fontSize+4,fontweight='normal',transform=ax1.transAxes)
ax2.text(-0.4,1.02,'E',fontsize=fontSize+4,fontweight='normal',transform=ax2.transAxes)

# make a diagram of the HoxB genes
# make a horizontal line above ax2
xShift = 1.3
textShift = 15
shift0 = -2.7
shift1 = -68 #-60
ax2.annotate('', xy=(0+xShift, 1.05+shift0), xycoords='axes fraction', xytext=(0+xShift, 2.3+shift0),
arrowprops=dict(arrowstyle="-", color='k', lw=2.0))
import matplotlib.patches as mpatches
# b7
rect = mpl.patches.Rectangle(
    (-0.05+xShift, 1.15+shift0), width=0.1, height=0.05, color="0.7", transform=ax2.transAxes,
    clip_on=False,zorder=10)
ax2.add_patch(rect)
ax2.text(150+textShift,22+shift1,'$\it{B7}$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
# b8
rect = mpl.patches.Rectangle(
    (-0.05+xShift, 1.35+shift0), width=0.1, height=0.05, color="0.7", transform=ax2.transAxes,
    clip_on=False,zorder=10)
ax2.add_patch(rect)
ax2.text(150+textShift,28+shift1,'$\it{B8}$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
# b9
rect = mpl.patches.Rectangle(
    (-0.05+xShift, 1.55+shift0), width=0.1, height=0.05, color="0.7", transform=ax2.transAxes,
    clip_on=False,zorder=10)
ax2.add_patch(rect)
ax2.text(150+textShift,34+shift1,'$\it{B9}$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
# b13
rect = mpl.patches.Rectangle(
    (-0.05+xShift, 2.15+shift0), width=0.1, height=0.05, color="0.7", transform=ax2.transAxes,
    clip_on=False,zorder=10)
ax2.add_patch(rect)
ax2.text(147+textShift,52.0+shift1,'$\it{B13}$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)

# draw lines from end of annotation to the plots in ax1 and ax2
# ax2
ax2.annotate('', xy=(-0.05+xShift, 1.6+shift0), xycoords='axes fraction', xytext=(0.0, 2.45+shift0),
arrowprops=dict(arrowstyle="-", color='k', lw=1.0))
ax2.annotate('', xy=(-0.075+xShift, 2.225+shift0), xycoords='axes fraction', xytext=(1.0, 2.45+shift0),
arrowprops=dict(arrowstyle="-", color='k', lw=1.0))
# for ax4
ax2.annotate('', xy=(0.125+xShift, 1.6+shift0), xycoords='axes fraction', xytext=(2.62, 2.45+shift0),
arrowprops=dict(arrowstyle="-", color='k', lw=1.0))
ax2.annotate('', xy=(0.05+xShift, 2.2+shift0), xycoords='axes fraction', xytext=(1.62, 2.45+shift0),
arrowprops=dict(arrowstyle="-", color='k', lw=1.0))
# and an arrow indicating that we are looking at the distance
ax2.annotate('', xy=(0.1+xShift, 1.6+shift0), xycoords='axes fraction', xytext=(0.1+xShift, 2.15+shift0),    
arrowprops=dict(arrowstyle="<->", color='k', lw=1.0))
ax2.text(90,-32,'$\it{HoxB}$\nintergenic\ndistance',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)


# plot the B9 distance and Lumbar
ax3.plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['L'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax3.plot((intergenicAll['B9'][intergenicAll[classKey]=='Aves'])/1000,intergenicAll['L'][intergenicAll[classKey]=='Aves'],marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax3.plot((intergenicAll['B9'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')])/1000,intergenicAll['L'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax3.set_ylabel('Lumbar',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[nameKey]!='snake']
y = intergenicAll['L'][intergenicAll[nameKey]!='snake']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
xPIC = intergenicLumbarPICB9['B9']
yPIC = intergenicLumbarPICB9['Lumbar']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
ax3.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax3.set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax3.set_xlim(-1,150)
ax3.set_xticks((0,50,100,150))
ax3.set_xticklabels([])
ax3.set_ylim(-1,15)
# set tick font size
ax3.tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('L vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'].to_list(),'B9','L',intergenicAll,tree,'_full')


# plot the B9 distance and Sacral
ax4.plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['S'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize)
ax4.plot((intergenicAll['B9'][intergenicAll[classKey]=='Aves'])/1000,intergenicAll['S'][intergenicAll[classKey]=='Aves'],marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize)
ax4.plot((intergenicAll['B9'][intergenicAll[classKey]=='Reptilia'])/1000,intergenicAll['S'][intergenicAll[classKey]=='Reptilia'],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize)
ax4.set_ylabel('Sacral',fontsize=fontSize)
ax4.set_xlabel('kb',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[nameKey]!='snake']
y = intergenicAll['S'][intergenicAll[nameKey]!='snake']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
xPIC = intergenicSacralPICB9['B9']
yPIC = intergenicSacralPICB9['Sacral']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)

# ax4.set_xlim(-1,250)
ax4.set_xlim(-1,150)
# ax4.set_xticks((0,100,200))
ax4.set_xticks((0,50,100,150))
ax4.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax4.set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax4.tick_params(axis='both', which='major', labelsize=fontSize)
print('S vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
# ax2.text(0.58,0.65,'tuatara',fontsize=fontSize,fontweight='normal',transform=ax2.transAxes)
saveForPIC(intergenicAll['Species'].to_list(),'B9','S',intergenicAll,tree,'_full')

# plot subplot labels ("a", "b", etc.)
ax3.text(-0.35,1.02,'F',fontsize=fontSize+4,fontweight='normal',transform=ax3.transAxes)
ax4.text(-0.38,1.02,'G',fontsize=fontSize+4,fontweight='normal',transform=ax4.transAxes)

# make a Hox diagram for ax0a
# delete the axes
ax0a.axis('off')

x0 = 0.3
shift2 = 0.2

ax0a.annotate('', xy=(x0+0*shift2, 0), xycoords='axes fraction', xytext=(x0+0*shift0, 1),
             arrowprops=dict(arrowstyle="-", color='k', lw=2.0))
ax0a.annotate('', xy=(x0+1*shift2, 0), xycoords='axes fraction', xytext=(x0+1*shift2, 1),
             arrowprops=dict(arrowstyle="-", color='k', lw=2.0))
ax0a.annotate('', xy=(x0+2*shift2, 0), xycoords='axes fraction', xytext=(x0+2*shift2, 1),
             arrowprops=dict(arrowstyle="-", color='k', lw=2.0))
ax0a.annotate('', xy=(x0+3*shift2, 0), xycoords='axes fraction', xytext=(x0+3*shift2, 1),
             arrowprops=dict(arrowstyle="-", color='k', lw=2.0))

# put an A,B,C, and D above each of these lines
ax0a.text(x0+0*shift2,1.025,'A',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+1*shift2,1.025,'B',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+2*shift2,1.025,'C',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+3*shift2,1.025,'D',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
# and below
ax0a.text(x0+0*shift2,-0.025,'A',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+1*shift2,-0.025,'B',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+2*shift2,-0.025,'C',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0+3*shift2,-0.025,'D',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)

# now need to put the "genes" on using rectangles
# for A we have all 1-13 except 8 and 12
# for B we're missing 10-12
# for C we're missing 2, 7
# for D we're missing 2, 5-7

geneHeight = 0.015
geneWidth = 0.08

# A 
genesA = [1,2,3,4,5,6,7,9,10,11,13]
for i in range(len(genesA)):
    rect = mpl.patches.Rectangle(
        (x0+0*shift2-0.5*geneWidth, (genesA[i]-0.5)/13 - 0.5*geneHeight), width=geneWidth, height=geneHeight, color="0.8", transform=ax0a.transAxes,
        clip_on=False,zorder=10)
    ax0a.add_patch(rect)
    
# B
genesB = [1,2,3,4,5,6,7,8,9,13]
for i in range(len(genesB)):
    rect = mpl.patches.Rectangle(
        (x0+1*shift2-0.5*geneWidth, (genesB[i]-0.5)/13 - 0.5*geneHeight), width=geneWidth, height=geneHeight, color="0.7", transform=ax0a.transAxes,
        clip_on=False,zorder=10)
    ax0a.add_patch(rect)
    
# C
genesC = [1,3,4,5,6,8,9,10,11,12,13]
for i in range(len(genesC)):
    rect = mpl.patches.Rectangle(
        (x0+2*shift2-0.5*geneWidth, (genesC[i]-0.5)/13 - 0.5*geneHeight), width=geneWidth, height=geneHeight, color="0.6", transform=ax0a.transAxes,
        clip_on=False,zorder=10)
    ax0a.add_patch(rect)

# D
genesD = [1,3,4,8,9,10,11,12,13]
for i in range(len(genesD)):
    rect = mpl.patches.Rectangle(
        (x0+3*shift2-0.5*geneWidth, (genesD[i]-0.5)/13 - 0.5*geneHeight), width=geneWidth, height=geneHeight, color="0.5", transform=ax0a.transAxes,
        clip_on=False,zorder=10)
    ax0a.add_patch(rect)

# plot "1"-"13" on the left
for i in range(13):
    ax0a.text(x0-0.18,(i+0.5)/13,str(i+1),fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)

# put "cluster" on the bottom and "gene" on the left
ax0a.text(x0+1.5*shift2,-0.085,'$\it{Hox}$ cluster',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=0)
ax0a.text(x0-0.35,0.5,'$\it{Hox}$ gene',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)

# label (a)
ax0a.text(-0.2,1.01,'A',fontsize=fontSize+4,fontweight='normal',transform=ax0a.transAxes)

# save
plt.savefig(outputPath+'plots/genotypePhenotypePNAS_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/genotypePhenotypePNAS_v2.pdf',dpi=300,bbox_inches='tight')


#%% plot the correlation matrix for all 4 Classes separately

# Create a figure and a gridspec layout
fig = plt.figure(figsize=(13,6))#,constrained_layout=False)
fig.subplots_adjust(hspace=0.7,wspace=3.0)
gs = gridspec.GridSpec(2,8)#,hspace=30000000)
ax0 = plt.subplot(gs[:,0:2])
ax1 = plt.subplot(gs[:,2:4])
ax2 = plt.subplot(gs[:,4:6])
ax3 = plt.subplot(gs[:,6:8])

# plot the mammal correlation matrix
corrMatAllRearranged = np.zeros((int(len(vertebralCols)*4),int(len(intergenicCols)/4)))
for i in range(len(vertebralCols)):
    corrMatAllRearranged[4*i+0,:] = corrMatMammals[i,:int(len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+1,:] = corrMatMammals[i,int(len(intergenicCols)/4):int(len(intergenicCols)/2)]
    corrMatAllRearranged[4*i+2,:] = corrMatMammals[i,int(len(intergenicCols)/2):int(3*len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+3,:] = corrMatMammals[i,int(3*len(intergenicCols)/4):]
# plot the matrix
# put the origin at the bottom left
im = ax0.imshow(corrMatAllRearranged.T, cmap=colorMap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',origin='lower')

ax0.set_xticks(np.arange(len(vertebralCols))*4+1.5)
ax0.set_xticklabels(vertebralCols,rotation=0,y=-0.01,fontsize=fontSize)
ax0.set_yticks(np.arange(13)-0.0)
ax0.set_yticklabels(np.arange(13)+1,fontsize=fontSize)
ax0.set_ylabel('$\it{Hox}$ intergenic distance',fontsize=fontSize)
ax0.set_xlabel('vertebrae',fontsize=fontSize)

# plot vertical dashed lines to separate the plasticities
for i in range(len(vertebralCols)-1):
    ax0.plot([4*(i+1)-0.5,4*(i+1)-0.5],[-0.5,12.5],'k--',linewidth=lineWidth)

# put some ticks labels (no ticks) on the top for the Hox clusters A,B,C,D repeating 5 times (one for each vertebral column)
ax0b = ax0.twiny()
ax0b.set_xlim(ax0.get_xlim())
ax0b.set_xticks(np.arange(len(vertebralCols)*4)+0.0)
hoxLetters = (['A','B','C','D'])
hoxLettersLabels = []
for i in range(len(vertebralCols)):
    for j in range(4):
        hoxLettersLabels.append(hoxLetters[j])
# ax0b2.set_xticklabels(hoxLettersLabels,rotation=45,fontsize=fontSize,horizontalalignment='left')
ax0b.set_xticklabels(['']*len(hoxLettersLabels))
ax0.text(-.05,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.06,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.16,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.275,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0b.transAxes)
ax0.text(-.05+0.22,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.06+0.22,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.16+0.22,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.275+0.22,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(-.05+0.44,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.06+0.44,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.16+0.44,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.275+0.44,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(-.05+0.66,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.06+0.66,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.16+0.66,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.275+0.66,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(-.05+0.88,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.06+0.88,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.16+0.88,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)
ax0.text(.275+0.88,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax0.transAxes)

# add the colorbar to the left
cbar = fig.colorbar(im, ax=ax0, orientation='vertical',pad=0.035,shrink=0.3,aspect=30)
# cbar.set_label('Pearson correlation coefficient', rotation=90,fontsize=fontSize)
# put the cbar label on top
cbar.set_label('r', rotation=0,fontsize=fontSize,y=1.15,labelpad=-20)
# set the ticks to be integers -1,0,1
cbar.set_ticks([-1,0,1])
# set the colorbar tick label font size
cbar.ax.tick_params(labelsize=fontSize)

# plot the bird correlation matrix
corrMatAllRearranged = np.zeros((int(len(vertebralCols)*4),int(len(intergenicCols)/4)))
for i in range(len(vertebralCols)):
    corrMatAllRearranged[4*i+0,:] = corrMatBirds[i,:int(len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+1,:] = corrMatBirds[i,int(len(intergenicCols)/4):int(len(intergenicCols)/2)]
    corrMatAllRearranged[4*i+2,:] = corrMatBirds[i,int(len(intergenicCols)/2):int(3*len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+3,:] = corrMatBirds[i,int(3*len(intergenicCols)/4):]
# plot the matrix
# put the origin at the bottom left
im = ax1.imshow(corrMatAllRearranged.T, cmap=colorMap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',origin='lower')

ax1.set_xticks(np.arange(len(vertebralCols))*4+1.5)
ax1.set_xticklabels(vertebralCols,rotation=0,y=-0.01,fontsize=fontSize)
ax1.set_yticks(np.arange(13)-0.0)
ax1.set_yticklabels(np.arange(13)+1,fontsize=fontSize)
ax1.set_ylabel('$\it{Hox}$ intergenic distance',fontsize=fontSize)
ax1.set_xlabel('vertebrae',fontsize=fontSize)

# plot vertical dashed lines to separate the plasticities
for i in range(len(vertebralCols)-1):
    ax1.plot([4*(i+1)-0.5,4*(i+1)-0.5],[-0.5,12.5],'k--',linewidth=lineWidth)

# put some ticks labels (no ticks) on the top for the Hox clusters A,B,C,D repeating 5 times (one for each vertebral column)
ax1b = ax1.twiny()
ax1b.set_xlim(ax1.get_xlim())
ax1b.set_xticks(np.arange(len(vertebralCols)*4)+0.0)
hoxLetters = (['A','B','C','D'])
hoxLettersLabels = []
for i in range(len(vertebralCols)):
    for j in range(4):
        hoxLettersLabels.append(hoxLetters[j])
# ax1b2.set_xticklabels(hoxLettersLabels,rotation=45,fontsize=fontSize,horizontalalignment='left')
ax1b.set_xticklabels(['']*len(hoxLettersLabels))
ax1.text(-.05,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.06,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.16,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.275,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax1b.transAxes)
ax1.text(-.05+0.22,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.06+0.22,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.16+0.22,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.275+0.22,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(-.05+0.44,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.06+0.44,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.16+0.44,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.275+0.44,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(-.05+0.66,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.06+0.66,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.16+0.66,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.275+0.66,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(-.05+0.88,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.06+0.88,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.16+0.88,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)
ax1.text(.275+0.88,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax1.transAxes)

# add the colorbar to the left
cbar = fig.colorbar(im, ax=ax1, orientation='vertical',pad=0.035,shrink=0.3,aspect=30)
# cbar.set_label('Pearson correlation coefficient', rotation=90,fontsize=fontSize)
# put the cbar label on top
cbar.set_label('r', rotation=0,fontsize=fontSize,y=1.15,labelpad=-20)
# set the ticks to be integers -1,0,1
cbar.set_ticks([-1,0,1])
# set the colorbar tick label font size
cbar.ax.tick_params(labelsize=fontSize)

# plot the reptile correlation matrix
corrMatAllRearranged = np.zeros((int(len(vertebralCols)*4),int(len(intergenicCols)/4)))
for i in range(len(vertebralCols)):
    corrMatAllRearranged[4*i+0,:] = corrMatReptiles[i,:int(len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+1,:] = corrMatReptiles[i,int(len(intergenicCols)/4):int(len(intergenicCols)/2)]
    corrMatAllRearranged[4*i+2,:] = corrMatReptiles[i,int(len(intergenicCols)/2):int(3*len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+3,:] = corrMatReptiles[i,int(3*len(intergenicCols)/4):]
# plot the matrix
# put the origin at the bottom left
im = ax2.imshow(corrMatAllRearranged.T, cmap=colorMap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',origin='lower')

ax2.set_xticks(np.arange(len(vertebralCols))*4+1.5)
ax2.set_xticklabels(vertebralCols,rotation=0,y=-0.01,fontsize=fontSize)
ax2.set_yticks(np.arange(13)-0.0)
ax2.set_yticklabels(np.arange(13)+1,fontsize=fontSize)
ax2.set_ylabel('$\it{Hox}$ intergenic distance',fontsize=fontSize)
ax2.set_xlabel('vertebrae',fontsize=fontSize)

# plot vertical dashed lines to separate the plasticities
for i in range(len(vertebralCols)-1):
    ax2.plot([4*(i+1)-0.5,4*(i+1)-0.5],[-0.5,12.5],'k--',linewidth=lineWidth)

# put some ticks labels (no ticks) on the top for the Hox clusters A,B,C,D repeating 5 times (one for each vertebral column)
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(np.arange(len(vertebralCols)*4)+0.0)
hoxLetters = (['A','B','C','D'])
hoxLettersLabels = []
for i in range(len(vertebralCols)):
    for j in range(4):
        hoxLettersLabels.append(hoxLetters[j])
# ax2b2.set_xticklabels(hoxLettersLabels,rotation=45,fontsize=fontSize,horizontalalignment='left')
ax2b.set_xticklabels(['']*len(hoxLettersLabels))
ax2.text(-.05,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.06,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.16,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.275,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax2b.transAxes)
ax2.text(-.05+0.22,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.06+0.22,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.16+0.22,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.275+0.22,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(-.05+0.44,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.06+0.44,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.16+0.44,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.275+0.44,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(-.05+0.66,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.06+0.66,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.16+0.66,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.275+0.66,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(-.05+0.88,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.06+0.88,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.16+0.88,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)
ax2.text(.275+0.88,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax2.transAxes)

# add the colorbar to the left
cbar = fig.colorbar(im, ax=ax2, orientation='vertical',pad=0.035,shrink=0.3,aspect=30)
# cbar.set_label('Pearson correlation coefficient', rotation=90,fontsize=fontSize)
# put the cbar label on top
cbar.set_label('r', rotation=0,fontsize=fontSize,y=1.15,labelpad=-20)
# set the ticks to be integers -1,0,1
cbar.set_ticks([-1,0,1])
# set the colorbar tick label font size
cbar.ax.tick_params(labelsize=fontSize)

# plot the amphibian correlation matrix
corrMatAllRearranged = np.zeros((int(len(vertebralCols)*4),int(len(intergenicCols)/4)))
for i in range(len(vertebralCols)):
    corrMatAllRearranged[4*i+0,:] = corrMatAmphibians[i,:int(len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+1,:] = corrMatAmphibians[i,int(len(intergenicCols)/4):int(len(intergenicCols)/2)]
    corrMatAllRearranged[4*i+2,:] = corrMatAmphibians[i,int(len(intergenicCols)/2):int(3*len(intergenicCols)/4)]
    corrMatAllRearranged[4*i+3,:] = corrMatAmphibians[i,int(3*len(intergenicCols)/4):]
# plot the matrix
# put the origin at the bottom left
im = ax3.imshow(corrMatAllRearranged.T, cmap=colorMap, vmin=-1, vmax=1, aspect='auto', interpolation='nearest',origin='lower')

ax3.set_xticks(np.arange(len(vertebralCols))*4+1.5)
ax3.set_xticklabels(vertebralCols,rotation=0,y=-0.01,fontsize=fontSize)
ax3.set_yticks(np.arange(13)-0.0)
ax3.set_yticklabels(np.arange(13)+1,fontsize=fontSize)
ax3.set_ylabel('$\it{Hox}$ intergenic distance',fontsize=fontSize)
ax3.set_xlabel('vertebrae',fontsize=fontSize)

# plot vertical dashed lines to separate the plasticities
for i in range(len(vertebralCols)-1):
    ax3.plot([4*(i+1)-0.5,4*(i+1)-0.5],[-0.5,12.5],'k--',linewidth=lineWidth)

# put some ticks labels (no ticks) on the top for the Hox clusters A,B,C,D repeating 5 times (one for each vertebral column)
ax3b = ax3.twiny()
ax3b.set_xlim(ax3.get_xlim())
ax3b.set_xticks(np.arange(len(vertebralCols)*4)+0.0)
hoxLetters = (['A','B','C','D'])
hoxLettersLabels = []
for i in range(len(vertebralCols)):
    for j in range(4):
        hoxLettersLabels.append(hoxLetters[j])
# ax3b2.set_xticklabels(hoxLettersLabels,rotation=45,fontsize=fontSize,horizontalalignment='left')
ax3b.set_xticklabels(['']*len(hoxLettersLabels))
ax3.text(-.05,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.06,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.16,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.275,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax3b.transAxes)
ax3.text(-.05+0.22,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.06+0.22,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.16+0.22,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.275+0.22,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(-.05+0.44,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.06+0.44,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.16+0.44,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.275+0.44,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(-.05+0.66,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.06+0.66,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.16+0.66,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.275+0.66,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(-.05+0.88,1.015,'A',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.06+0.88,1.035,'B',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.16+0.88,1.055,'C',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)
ax3.text(.275+0.88,1.075,'D',fontsize=fontSize-1,fontweight='normal',transform=ax3.transAxes)

# add the colorbar to the left
cbar = fig.colorbar(im, ax=ax3, orientation='vertical',pad=0.035,shrink=0.3,aspect=30)
# cbar.set_label('Pearson correlation coefficient', rotation=90,fontsize=fontSize)
# put the cbar label on top
cbar.set_label('r', rotation=0,fontsize=fontSize,y=1.15,labelpad=-20)
# set the ticks to be integers -1,0,1
cbar.set_ticks([-1,0,1])
# set the colorbar tick label font size
cbar.ax.tick_params(labelsize=fontSize)

# plot subplot labels ("a", "b", etc.)
ax0.text(-.4,1.01,'A',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)
ax1.text(-.4,1.01,'B',fontsize=fontSize+4,fontweight='normal',transform=ax1.transAxes)
ax2.text(-0.4,1.02,'C',fontsize=fontSize+4,fontweight='normal',transform=ax2.transAxes)
ax3.text(-0.4,1.02,'D',fontsize=fontSize+4,fontweight='normal',transform=ax3.transAxes)

# put a subplot title on each one

ax0.set_title('Mammalia',fontsize=fontSize, y=1.15)
ax1.set_title('Aves',fontsize=fontSize, y=1.15)
ax2.set_title('Reptilia',fontsize=fontSize, y=1.15)
ax3.set_title('Amphibia',fontsize=fontSize, y=1.15)

# save
plt.savefig(outputPath+'plots/intergenicVertebralCorrelationsByClass_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/intergenicVertebralCorrelationsByClass_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')


#%% make another extended data figure with the ctcf experimental counts (need to load)
# and several other examples of vertebral counts vs. intergenic distances

fontSize = 12
fontToUse = 'Arial'
markerSize = 6
params = {
   'axes.labelsize': fontSize,
#   'font.family': 'sans-serif',
  # 'font.serif': 'Times New Roman',
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
#   'legend.fontsize': 11,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
#   'figure.figsize': [4.5, 4.5]
   }
mpl.rcParams.update(params)

# make a grid of 2 rows and 3 plots
fig,ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.flatten()
# put some more space between the subplots
plt.subplots_adjust(wspace=0.4,hspace=0.4)

ctcfCounts = pd.read_csv(inputPath+'ctcfExperimentalCounts.csv')
experimentalOrganisms = ctcfCounts['organism'].unique()

i = 0
ax[0].plot(ctcfCounts['hoxLen'][ctcfCounts['organism']==experimentalOrganisms[i]],ctcfCounts['peakNum'][ctcfCounts['organism']==experimentalOrganisms[i]],marker=markerWheel[0],color=colorWheel[0],linestyle='None',label=experimentalOrganisms[i])
i = 1
ax[0].plot(ctcfCounts['hoxLen'][ctcfCounts['organism']==experimentalOrganisms[i]],ctcfCounts['peakNum'][ctcfCounts['organism']==experimentalOrganisms[i]],marker=markerWheel[0],color=colorWheel[4],linestyle='None',label=experimentalOrganisms[i])
i = 2
ax[0].plot(ctcfCounts['hoxLen'][ctcfCounts['organism']==experimentalOrganisms[i]],ctcfCounts['peakNum'][ctcfCounts['organism']==experimentalOrganisms[i]],marker=markerWheel[1],color=colorWheel[1],linestyle='None',label=experimentalOrganisms[i])
i = 3
ax[0].plot(ctcfCounts['hoxLen'][ctcfCounts['organism']==experimentalOrganisms[i]],ctcfCounts['peakNum'][ctcfCounts['organism']==experimentalOrganisms[i]],marker=markerWheel[3],color=colorWheel[3],linestyle='None',label=experimentalOrganisms[i])
ax[0].set_xlabel('hox cluster length (bp)',fontsize=fontSize)
ax[0].set_ylabel('number of peaks',fontsize=fontSize)
r,p = scipy.stats.pearsonr(ctcfCounts['hoxLen'],ctcfCounts['peakNum'])
r,p = scipy.stats.pearsonr(ctcfCounts['hoxLen'],ctcfCounts['peakNum'])
ax[0].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p),fontsize=fontSize)

ax[0].annotate('human',
    xy=(0.94,0.30), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right',fontsize=fontSize-1)
ax[0].annotate('mouse',
    xy=(0.94,0.22), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[4],ha='right',fontsize=fontSize-1)
ax[0].annotate('chicken',
    xy=(0.94,0.14), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right',fontsize=fontSize-1)
ax[0].annotate('frog',
    xy=(0.94,0.06), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[3],ha='right',fontsize=fontSize-1)

# plot mammalian Caudal vs. intergenic distance B9-B13
ax[1].plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['Ca'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[1].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
ax[1].set_ylabel('Caudal',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[classKey]=='Mammalia']
y = intergenicAll['Ca'][intergenicAll[classKey]=='Mammalia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
# determine which of intergenicCervicalPIC are only mammalia
# this can be done by finding any in column classKey that have either 'Aves', 'Reptilia', or 'Amphibia'
intergenicCaudalPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Caudal_B9'+suffix+'.csv')
mammalOnlyCaudalPIC = intergenicCaudalPICB9[intergenicCaudalPICB9[classKey].apply(lambda x: ('Aves' not in x) & ('Reptilia' not in x) & ('Amphibia' not in x))]
xPIC = mammalOnlyCaudalPIC['B9']
yPIC = mammalOnlyCaudalPIC['Caudal']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
ax[1].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[1].set_title('r='+str(np.round(r,2))+', PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[1].set_xlim(40,130)
ax[1].set_xticks((50,100))
ax[1].set_ylim(0,32)
ax[1].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('Ca vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax[1].text(0.65,0.91,'Mammalia',fontsize=fontSize,transform=ax[1].transAxes)
saveForPIC(intergenicAll['Species'][intergenicAll[classKey]=='Mammalia'].to_list(),'B9','Ca',intergenicAll,tree,'_mammals')

# circle the Cavia porcellus (guineaPig) and the other small tail species and put their name in text
additionalCircleSize = 7
ax[1].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[1].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000+1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus']+1.2,'$\it{Cavia}$\n$\it{porcellus}$',fontsize=fontSize-2,ha='left',va='bottom')
# now the others, which are Homo sapiens, Pan troglodytes, and Choloepus didactylus
ax[1].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[1].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000+2,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens']+0.5,'$\it{Homo}$\n$\it{sapiens}$',fontsize=fontSize-2,ha='left',va='bottom')
ax[1].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[1].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes']+0.25,'$\it{Pan}$ $\it{troglodytes}$',fontsize=fontSize-2,ha='right',va='top')
ax[1].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[1].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000-3,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],'$\it{Choloepus}$\n$\it{didactylus}$',fontsize=fontSize-2,ha='right',va='bottom')
# and the mouse
ax[1].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[1].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],'$\it{Mus}$\n$\it{musculus}$',fontsize=fontSize-2,ha='right',va='top')

# plot Cervical vs. B4 for reptiles
ax[2].plot((intergenicAll['B4'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')])/1000,intergenicAll['C'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax[2].text(0.05,0.05,'Reptilia',fontsize=fontSize,transform=ax[2].transAxes)
ax[2].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[2].set_ylabel('Cervical',fontsize=fontSize)
x = intergenicAll['B4'][intergenicAll[classKey]=='Reptilia']
y = intergenicAll['C'][intergenicAll[classKey]=='Reptilia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
# determine which of intergenicCervicalPIC are only mammalia
# this can be done by finding any in column classKey that have either 'Aves', 'Reptilia', or 'Amphibia'
intergenicCervicalPICB4 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Cervical_B4'+suffix+'.csv')
reptileOnlyCervicalPICB4 = intergenicCervicalPICB4[intergenicCervicalPICB4[classKey].apply(lambda x: ('Aves' not in x) & ('Mammalia' not in x) & ('Amphibia' not in x))]
xPIC = reptileOnlyCervicalPICB4['B4']
yPIC = reptileOnlyCervicalPICB4['Cervical']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
# ax[2].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[2].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[2].set_xlim(10,30)
ax[2].set_ylim(0,10)
# set tick font size
ax[2].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'][intergenicAll[classKey]=='Reptilia'].to_list(),'B4','C',intergenicAll,tree,'_reptiles')

# mark a turtle
ax[2].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[2].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas']+0.4,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[2].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[2].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara']+0.4,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# plot Thoracic vs. the same intergenic distances
# no inset
ax[3].plot((intergenicAll['B4'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')])/1000,intergenicAll['T'][(intergenicAll[classKey]=='Reptilia')&(intergenicAll[nameKey]!='snake')],marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax[3].text(0.72,0.05,'Reptilia',fontsize=fontSize,transform=ax[3].transAxes)
ax[3].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[3].set_ylabel('Thoracic',fontsize=fontSize)
x = intergenicAll['B4'][intergenicAll[classKey]=='Reptilia']
y = intergenicAll['T'][intergenicAll[classKey]=='Reptilia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
# determine which of intergenicCervicalPIC are only mammalia
# this can be done by finding any in column classKey that have either 'Aves', 'Reptilia', or 'Amphibia'
intergenicThoracicPICB4 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Thoracic_B4'+suffix+'.csv')
reptileOnlyThoracicPICB4 = intergenicThoracicPICB4[intergenicThoracicPICB4[classKey].apply(lambda x: ('Aves' not in x) & ('Mammalia' not in x) & ('Amphibia' not in x))]
xPIC = reptileOnlyThoracicPICB4['B4']
yPIC = reptileOnlyThoracicPICB4['Thoracic']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
# ax[3].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[3].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[3].set_xlim(10,30)
ax[3].set_ylim(5,30)
# set tick font size
ax[3].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'][intergenicAll[classKey]=='Reptilia'].to_list(),'B4','T',intergenicAll,tree,'_reptiles')

# mark a turtle
ax[3].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[3].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas']+1,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[3].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[3].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara']-1,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# Thoracic vs. D1 for amphibians
ax[4].plot((intergenicAll['D1'][(intergenicAll[classKey]=='Amphibia')&(intergenicAll[nameKey]!='snake')])/1000,intergenicAll['T'][(intergenicAll[classKey]=='Amphibia')&(intergenicAll[nameKey]!='snake')],marker='>',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax[4].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[4].transAxes)
ax[4].set_xlabel('intergenic distance\n$\it{D1}$-$\it{D3}$ (kb)',fontsize=fontSize)
ax[4].set_ylabel('Thoracic',fontsize=fontSize)
x = intergenicAll['D1'][intergenicAll[classKey]=='Amphibia']
y = intergenicAll['T'][intergenicAll[classKey]=='Amphibia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
# determine which of intergenicCervicalPIC are only mammalia
# this can be done by finding any in column classKey that have either 'Aves', 'Reptilia', or 'Amphibia'
intergenicThoracicPICD1 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Thoracic_D1'+suffix+'.csv')
amphibianOnlyThoracicPICD1 = intergenicThoracicPICD1[intergenicThoracicPICD1[classKey].apply(lambda x: ('Aves' not in x) & ('Mammalia' not in x) & ('Amphibia' not in x))]
xPIC = amphibianOnlyThoracicPICD1['D1']
yPIC = amphibianOnlyThoracicPICD1['Thoracic']
rPIC,pPIC = scipy.stats.pearsonr(xPIC[np.isfinite(yPIC)],yPIC[np.isfinite(yPIC)])
# ax[4].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'**',fontsize=fontSize)
ax[4].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[4].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'][intergenicAll[classKey]=='Amphibia'].to_list(),'D1','T',intergenicAll,tree,'_amphibians')

# mark a frog
ax[4].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis']+0.35,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[4].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-0.25,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[4].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl']-0.25,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# and finally amphibian Ca vs. A11
ax[5].plot(intergenicAll['A11'][intergenicAll[classKey]=='Amphibia']/1000,intergenicAll['Ca'][intergenicAll[classKey]=='Amphibia'],marker='>',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
ax[5].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[5].transAxes)
ax[5].set_xlabel('intergenic distance\n$\it{A11}$-$\it{A12}$ (kb)',fontsize=fontSize)
ax[5].set_ylabel('Caudal',fontsize=fontSize)
x = intergenicAll['A11'][intergenicAll[classKey]=='Amphibia']
y = intergenicAll['Ca'][intergenicAll[classKey]=='Amphibia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
intergenicCaudalPICA11 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Caudal_A11'+suffix+'.csv')
amphibianOnlyCaudalPICA11 = intergenicCaudalPICA11[intergenicCaudalPICA11[classKey].apply(lambda x: ('Aves' not in x) & ('Mammalia' not in x) & ('Amphibia' not in x))]
xPIC = amphibianOnlyCaudalPICA11['A11']
yPIC = amphibianOnlyCaudalPICA11['Caudal']
rPIC,pPIC = scipy.stats.pearsonr(xPIC[np.isfinite(yPIC)],yPIC[np.isfinite(yPIC)])
# ax[5].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[5].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[5].set_xlim(0,70)
# set tick font size
ax[5].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. A2+D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'][intergenicAll[classKey]=='Amphibia'].to_list(),'A11','Ca',intergenicAll,tree,'_amphibians')

# mark a frog
ax[5].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis']+1,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[5].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-1,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[5].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl']-1,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# subplot labels
ax[0].text(-0.2,1.05,'A',fontsize=fontSize+4,fontweight='normal',transform=ax[0].transAxes)
ax[1].text(-0.2,1.05,'B',fontsize=fontSize+4,fontweight='normal',transform=ax[1].transAxes)
ax[2].text(-0.2,1.05,'C',fontsize=fontSize+4,fontweight='normal',transform=ax[2].transAxes)
ax[3].text(-0.2,1.05,'D',fontsize=fontSize+4,fontweight='normal',transform=ax[3].transAxes)
ax[4].text(-0.2,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax[4].transAxes)
ax[5].text(-0.2,1.05,'F',fontsize=fontSize+4,fontweight='normal',transform=ax[5].transAxes)

# save
plt.savefig(outputPath+'plots/experimentalCTCFAndIntergenicExamples_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/experimentalCTCFAndIntergenicExamples_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')


#%% additionally make a plot comparing the NCBI annotated genome results to those annotated by us using BLAST

fontSize = 12
fontToUse = 'Arial'
markerSize = 6
params = {
   'axes.labelsize': fontSize,
#   'font.family': 'sans-serif',
  # 'font.serif': 'Times New Roman',
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
#   'legend.fontsize': 11,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
#   'figure.figsize': [4.5, 4.5]
   }
mpl.rcParams.update(params)

# make a grid of 3 rows and 3 plots
fig,ax = plt.subplots(3,3,figsize=(12,12.25))
ax = ax.flatten()
# put some more space between the subplots
plt.subplots_adjust(wspace=0.4,hspace=0.5)

# plot the B9 distance and Cervical
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))]
xAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yAnnotatedMammals = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yAnnotatedAves = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B9','C']),speciesKey,nameKey,classKey)
xNotAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotAnnotatedMammals = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotAnnotatedAves = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[0].plot((xAnnotatedMammals)/1000,yAnnotatedMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[0].plot((xAnnotatedAves)/1000,yAnnotatedAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[0].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[0].plot((xNotAnnotatedMammals)/1000,yNotAnnotatedMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].plot((xNotAnnotatedAves)/1000,yNotAnnotatedAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].set_ylabel('Cervical',fontsize=fontSize)
ax[0].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xAnnotatedMammals,xAnnotatedAves,xAnnotatedReptilia))
y = np.concatenate((yAnnotatedMammals,yAnnotatedAves,yAnnotatedReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['C'])
ax[0].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[0].set_xlim(-1,150)
ax[0].set_xticks((0,50,100,150))
ax[0].set_xticklabels([])
ax[0].set_ylim(0,25)
# set tick font size
ax[0].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax[0].annotate('Mammalia',
    xy=(0.94,0.85), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right',fontsize=fontSize-1)
ax[0].annotate('Aves',
    xy=(0.94,0.72), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right',fontsize=fontSize-1)
ax[0].annotate('Reptilia',
    xy=(0.94,0.59), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[2],ha='right',fontsize=fontSize-1)

# plot the B9 distance and Thoracic
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))]
xAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yAnnotatedMammals = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yAnnotatedAves = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B9','T']),speciesKey,nameKey,classKey)
xNotAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotAnnotatedMammals = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotAnnotatedAves = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[1].plot((xAnnotatedMammals)/1000,yAnnotatedMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[1].plot((xAnnotatedAves)/1000,yAnnotatedAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[1].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[1].plot((xNotAnnotatedMammals)/1000,yNotAnnotatedMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[1].plot((xNotAnnotatedAves)/1000,yNotAnnotatedAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[1].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[1].set_ylabel('Thoracic',fontsize=fontSize)
ax[1].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xAnnotatedMammals,xAnnotatedAves,xAnnotatedReptilia))
y = np.concatenate((yAnnotatedMammals,yAnnotatedAves,yAnnotatedReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['T'])

ax[1].set_yticks((0,10,20,30))
# ax[1].set_xlim(-1,250)
ax[1].set_xlim(-1,150)
# ax[1].set_xticks((0,100,200))
ax[1].set_xticks((0,50,100,150))
ax[1].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[1].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'*',fontsize=fontSize)
ax[1].tick_params(axis='both', which='major', labelsize=fontSize)
print('T vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
# ax2.text(0.58,0.65,'tuatara',fontsize=fontSize,fontweight='normal',transform=ax2.transAxes)


# plot the B9 distance and Lumbar
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))]
xAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yAnnotatedMammals = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yAnnotatedAves = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B9','L']),speciesKey,nameKey,classKey)
xNotAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotAnnotatedMammals = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotAnnotatedAves = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[2].plot((xAnnotatedMammals)/1000,yAnnotatedMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[2].plot((xAnnotatedAves)/1000,yAnnotatedAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[2].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[2].plot((xNotAnnotatedMammals)/1000,yNotAnnotatedMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].plot((xNotAnnotatedAves)/1000,yNotAnnotatedAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].set_ylabel('Lumbar',fontsize=fontSize)
ax[2].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xAnnotatedMammals,xAnnotatedAves,xAnnotatedReptilia))
y = np.concatenate((yAnnotatedMammals,yAnnotatedAves,yAnnotatedReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['L'])
ax[2].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[2].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[2].set_xlim(-1,150)
ax[2].set_xticks((0,50,100,150))
ax[2].set_xticklabels([])
ax[2].set_ylim(-1,15)
# set tick font size
ax[2].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('L vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'].to_list(),'B9','L',intergenicAll,tree,'_full')

# plot the B9 distance and Sacral
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))]
xAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yAnnotatedMammals = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yAnnotatedAves = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B9','S']),speciesKey,nameKey,classKey)
xNotAnnotatedMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotAnnotatedMammals = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotAnnotatedAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotAnnotatedAves = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotAnnotatedReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[3].plot((xAnnotatedMammals)/1000,yAnnotatedMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[3].plot((xAnnotatedAves)/1000,yAnnotatedAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[3].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[3].plot((xNotAnnotatedMammals)/1000,yNotAnnotatedMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[3].plot((xNotAnnotatedAves)/1000,yNotAnnotatedAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[3].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[3].set_ylabel('Sacral',fontsize=fontSize)
ax[3].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xAnnotatedMammals,xAnnotatedAves,xAnnotatedReptilia))
y = np.concatenate((yAnnotatedMammals,yAnnotatedAves,yAnnotatedReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['S'])

# ax[3].set_xlim(-1,250)
ax[3].set_xlim(-1,150)
# ax[3].set_xticks((0,100,200))
ax[3].set_xticks((0,50,100,150))
ax[3].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[3].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[3].tick_params(axis='both', which='major', labelsize=fontSize)
print('S vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# plot mammalian Caudal vs. intergenic distance B9-B13
# All mammalian genomes we used were already annotated!
ax[4].plot((intergenicAll['B9'][intergenicAll[classKey]=='Mammalia'])/1000,intergenicAll['Ca'][intergenicAll[classKey]=='Mammalia'],marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[4].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
ax[4].set_ylabel('Caudal',fontsize=fontSize)
x = intergenicAll['B9'][intergenicAll[classKey]=='Mammalia']
y = intergenicAll['Ca'][intergenicAll[classKey]=='Mammalia']
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
# determine which of intergenicCervicalPIC are only mammalia
# this can be done by finding any in column classKey that have either 'Aves', 'Reptilia', or 'Amphibia'
intergenicCaudalPICB9 = pd.read_csv(inputPath2+'intergenicIndividual/pic_normalized_intergenic_Caudal_B9'+suffix+'.csv')
mammalOnlyCaudalPIC = intergenicCaudalPICB9[intergenicCaudalPICB9[classKey].apply(lambda x: ('Aves' not in x) & ('Reptilia' not in x) & ('Amphibia' not in x))]
xPIC = mammalOnlyCaudalPIC['B9']
yPIC = mammalOnlyCaudalPIC['Caudal']
rPIC,pPIC = scipy.stats.pearsonr(xPIC,yPIC)
ax[4].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[4].set_title('r='+str(np.round(r,2))+', PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[4].set_xlim(40,130)
ax[4].set_xticks((50,100))
ax[4].set_ylim(0,32)
ax[4].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('Ca vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax[4].text(0.65,0.91,'Mammalia',fontsize=fontSize,transform=ax[4].transAxes)

# circle the Cavia porcellus (guineaPig) and the other small tail species and put their name in text
additionalCircleSize = 7
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000+1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus']+1.2,'$\it{Cavia}$\n$\it{porcellus}$',fontsize=fontSize-2,ha='left',va='bottom')
# now the others, which are Homo sapiens, Pan troglodytes, and Choloepus didactylus
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000+2,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens']+0.5,'$\it{Homo}$\n$\it{sapiens}$',fontsize=fontSize-2,ha='left',va='bottom')
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes']+0.25,'$\it{Pan}$ $\it{troglodytes}$',fontsize=fontSize-2,ha='right',va='top')
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000-3,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],'$\it{Choloepus}$\n$\it{didactylus}$',fontsize=fontSize-2,ha='right',va='bottom')
# and the mouse
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],'$\it{Mus}$\n$\it{musculus}$',fontsize=fontSize-2,ha='right',va='top')

# plot Cervical vs. B4 for reptiles
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xAnnotatedReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B4','C']),speciesKey,nameKey,classKey)
xNotAnnotatedReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[5].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[5].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[5].text(0.05,0.05,'Reptilia',fontsize=fontSize,transform=ax[5].transAxes)
ax[5].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[5].set_ylabel('Cervical',fontsize=fontSize)
x = xAnnotatedReptilia
y = yAnnotatedReptilia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B4'],xPICn['C'])
# ax[5].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[5].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[5].set_xlim(10,30)
ax[5].set_ylim(0,10)
# set tick font size
ax[5].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a turtle
ax[5].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas']+0.4,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[5].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara']+0.4,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# plot Thoracic vs. the same intergenic distances
# no inset
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xAnnotatedReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yAnnotatedReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['B4','T']),speciesKey,nameKey,classKey)
xNotAnnotatedReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yNotAnnotatedReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[6].plot((xAnnotatedReptilia)/1000,yAnnotatedReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[6].plot((xNotAnnotatedReptilia)/1000,yNotAnnotatedReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[6].text(0.72,0.05,'Reptilia',fontsize=fontSize,transform=ax[6].transAxes)
ax[6].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[6].set_ylabel('Thoracic',fontsize=fontSize)
x = xAnnotatedReptilia
y = yAnnotatedReptilia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B4'],xPICn['T'])
# ax[6].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[6].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[6].set_xlim(10,30)
ax[6].set_ylim(5,30)
# set tick font size
ax[6].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a turtle
ax[6].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[6].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas']+1,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[6].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[6].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara']-1,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# Thoracic vs. D1 for amphibians
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
xAnnotatedAmphibia = intergenicAll['D1'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
yAnnotatedAmphibia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['D1','T']),speciesKey,nameKey,classKey)
xNotAnnotatedAmphibia = intergenicAll['D1'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
yNotAnnotatedAmphibia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
markerSize = 6
ax[7].plot((xAnnotatedAmphibia)/1000,yAnnotatedAmphibia,marker='^',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[7].plot((xNotAnnotatedAmphibia)/1000,yNotAnnotatedAmphibia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[7].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[7].transAxes)
ax[7].set_xlabel('intergenic distance\n$\it{D1}$-$\it{D3}$ (kb)',fontsize=fontSize)
ax[7].set_ylabel('Thoracic',fontsize=fontSize)
x = xAnnotatedAmphibia
y = yAnnotatedAmphibia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['D1'],xPICn['T'])
# ax[7].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'**',fontsize=fontSize)
# ax[7].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[7].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a frog
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis']+0.35,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-0.25,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl']-0.25,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# and finally amphibian Ca vs. A11
speciesAnnotated = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
xAnnotatedAmphibia = intergenicAll['A11'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
yAnnotatedAmphibia = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesAnnotated,(['A11','Ca']),speciesKey,nameKey,classKey)
xNotAnnotatedAmphibia = intergenicAll['A11'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
yNotAnnotatedAmphibia = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Annotated']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
markerSize = 6
ax[8].plot((xAnnotatedAmphibia)/1000,yAnnotatedAmphibia,marker='^',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[8].plot((xNotAnnotatedAmphibia)/1000,yNotAnnotatedAmphibia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
x = xAnnotatedAmphibia
y = yAnnotatedAmphibia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['A11'],xPICn['Ca'])
ax[8].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[8].transAxes)
ax[8].set_xlabel('intergenic distance\n$\it{A11}$-$\it{A12}$ (kb)',fontsize=fontSize)
ax[8].set_ylabel('Caudal',fontsize=fontSize)
# ax[8].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
# ax[8].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[8].set_xlim(0,70)
# set tick font size
ax[8].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. A2+D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a frog
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis']+1,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-1,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl']-1,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# subplot labels
ax[0].text(-0.2,1.05,'A',fontsize=fontSize+4,fontweight='normal',transform=ax[0].transAxes)
ax[1].text(-0.2,1.05,'B',fontsize=fontSize+4,fontweight='normal',transform=ax[1].transAxes)
ax[2].text(-0.2,1.05,'C',fontsize=fontSize+4,fontweight='normal',transform=ax[2].transAxes)
ax[3].text(-0.2,1.05,'D',fontsize=fontSize+4,fontweight='normal',transform=ax[3].transAxes)
ax[4].text(-0.2,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax[4].transAxes)
ax[5].text(-0.2,1.05,'F',fontsize=fontSize+4,fontweight='normal',transform=ax[5].transAxes)
ax[6].text(-0.2,1.05,'G',fontsize=fontSize+4,fontweight='normal',transform=ax[6].transAxes)
ax[7].text(-0.2,1.05,'H',fontsize=fontSize+4,fontweight='normal',transform=ax[7].transAxes)
ax[8].text(-0.2,1.05,'I',fontsize=fontSize+4,fontweight='normal',transform=ax[8].transAxes)

# save
plt.savefig(outputPath+'plots/intergenicAnnotatedOnly_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/intergenicAnnotatedOnly_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')


#%% additionally make a plot using filters on the genomes with high thresholds

annotation = 'yes'
coverage = 20
busco = 95
# make a "BUSCO coverage" column from "BUSCO score" taking the numerical value in between "C:" and "%"
# if the "BUSCO score" is "-", then set this value to zero
intergenicAll['BUSCO coverage'] = [float(x.split('C:')[1].split('%')[0]) if x!='-' else 0 for x in intergenicAll['BUSCO score']]

# make sure the "Coverage" column is a float
# first set any '-' to NaN
intergenicAll['Coverage'] = [np.nan if x=='-' else x for x in intergenicAll['Coverage']]
intergenicAll['Coverage'] = intergenicAll['Coverage'].astype(float)

# if we meet all the above criterion, make a new column called "Kept"
# but some of these have no BUSCO score, so we need to check for that as well
intergenicAll['Kept'] = ['yes' if x==annotation and y>=coverage and z>=busco else 'no' for x,y,z in zip(intergenicAll['Annotated'],intergenicAll['Coverage'],intergenicAll['BUSCO coverage'])]


fontSize = 12
fontToUse = 'Arial'
markerSize = 6
params = {
   'axes.labelsize': fontSize,
#   'font.family': 'sans-serif',
  # 'font.serif': 'Times New Roman',
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
#   'legend.fontsize': 11,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
#   'figure.figsize': [4.5, 4.5]
   }
mpl.rcParams.update(params)

# make a grid of 3 rows and 3 plots
fig,ax = plt.subplots(3,3,figsize=(12,12.25))
ax = ax.flatten()
# put some more space between the subplots
plt.subplots_adjust(wspace=0.4,hspace=0.5)

# plot the B9 distance and Cervical
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))]
xKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yKeptMammals = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yKeptAves = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B9','C']),speciesKey,nameKey,classKey)
xNotKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotKeptMammals = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotKeptAves = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[0].plot((xKeptMammals)/1000,yKeptMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[0].plot((xKeptAves)/1000,yKeptAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[0].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[0].plot((xNotKeptMammals)/1000,yNotKeptMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].plot((xNotKeptAves)/1000,yNotKeptAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[0].set_ylabel('Cervical',fontsize=fontSize)
ax[0].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xKeptMammals,xKeptAves,xKeptReptilia))
y = np.concatenate((yKeptMammals,yKeptAves,yKeptReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['C'])
ax[0].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[0].set_xlim(-1,150)
ax[0].set_xticks((0,50,100,150))
ax[0].set_xticklabels([])
ax[0].set_ylim(0,25)
# set tick font size
ax[0].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax[0].annotate('Mammalia',
    xy=(0.94,0.85), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right',fontsize=fontSize-1)
ax[0].annotate('Aves',
    xy=(0.94,0.72), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right',fontsize=fontSize-1)
ax[0].annotate('Reptilia',
    xy=(0.94,0.59), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[2],ha='right',fontsize=fontSize-1)

# plot the B9 distance and Thoracic
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))]
xKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yKeptMammals = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yKeptAves = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B9','T']),speciesKey,nameKey,classKey)
xNotKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotKeptMammals = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotKeptAves = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[1].plot((xKeptMammals)/1000,yKeptMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[1].plot((xKeptAves)/1000,yKeptAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[1].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[1].plot((xNotKeptMammals)/1000,yNotKeptMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[1].plot((xNotKeptAves)/1000,yNotKeptAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[1].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[1].set_ylabel('Thoracic',fontsize=fontSize)
ax[1].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xKeptMammals,xKeptAves,xKeptReptilia))
y = np.concatenate((yKeptMammals,yKeptAves,yKeptReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['T'])

ax[1].set_yticks((0,10,20,30))
# ax[1].set_xlim(-1,250)
ax[1].set_xlim(-1,150)
# ax[1].set_xticks((0,100,200))
ax[1].set_xticks((0,50,100,150))
ax[1].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[1].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'*',fontsize=fontSize)
ax[1].tick_params(axis='both', which='major', labelsize=fontSize)
print('T vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
# ax2.text(0.58,0.65,'tuatara',fontsize=fontSize,fontweight='normal',transform=ax2.transAxes)


# plot the B9 distance and Lumbar
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))]
xKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yKeptMammals = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yKeptAves = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B9','L']),speciesKey,nameKey,classKey)
xNotKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotKeptMammals = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotKeptAves = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['L'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['L'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[2].plot((xKeptMammals)/1000,yKeptMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[2].plot((xKeptAves)/1000,yKeptAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[2].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[2].plot((xNotKeptMammals)/1000,yNotKeptMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].plot((xNotKeptAves)/1000,yNotKeptAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[2].set_ylabel('Lumbar',fontsize=fontSize)
ax[2].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xKeptMammals,xKeptAves,xKeptReptilia))
y = np.concatenate((yKeptMammals,yKeptAves,yKeptReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['L'])
ax[2].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[2].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[2].set_xlim(-1,150)
ax[2].set_xticks((0,50,100,150))
ax[2].set_xticklabels([])
ax[2].set_ylim(-1,15)
# set tick font size
ax[2].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('L vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
saveForPIC(intergenicAll['Species'].to_list(),'B9','L',intergenicAll,tree,'_full')

# plot the B9 distance and Sacral
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))]
xKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yKeptMammals = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yKeptAves = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B9','S']),speciesKey,nameKey,classKey)
xNotKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotKeptMammals = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xNotKeptAves = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
yNotKeptAves = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Aves')]
xNotKeptReptilia = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['S'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['S'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[3].plot((xKeptMammals)/1000,yKeptMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
ax[3].plot((xKeptAves)/1000,yKeptAves,marker='s',color=colorWheel[1],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{B}}$')
ax[3].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[3].plot((xNotKeptMammals)/1000,yNotKeptMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[3].plot((xNotKeptAves)/1000,yNotKeptAves,marker='s',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[3].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[3].set_ylabel('Sacral',fontsize=fontSize)
ax[3].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
x = np.concatenate((xKeptMammals,xKeptAves,xKeptReptilia))
y = np.concatenate((yKeptMammals,yKeptAves,yKeptReptilia))
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['S'])

# ax[3].set_xlim(-1,250)
ax[3].set_xlim(-1,150)
# ax[3].set_xticks((0,100,200))
ax[3].set_xticks((0,50,100,150))
ax[3].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[3].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[3].tick_params(axis='both', which='major', labelsize=fontSize)
print('S vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# plot mammalian Caudal vs. intergenic distance B9-B13
# All mammalian genomes we used were already annotated!
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['B9']))]
xKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yKeptMammals = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B9','Ca']),speciesKey,nameKey,classKey)
xNotKeptMammals = intergenicAll['B9'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
yNotKeptMammals = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['B9']))&(intergenicAll[classKey]=='Mammalia')]
markerSize = 6
ax[4].plot((xKeptMammals)/1000,yKeptMammals,marker='o',color=colorWheel[0],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{M}}$')
# plot the not annotated as empty symbols
ax[4].plot((xNotKeptMammals)/1000,yNotKeptMammals,marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
ax[4].set_xlabel('intergenic distance\n$\it{B9}$-$\it{B13}$ (kb)',fontsize=fontSize)
ax[4].set_ylabel('Caudal',fontsize=fontSize)
x = xKeptMammals
y = yKeptMammals
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B9'],xPICn['Ca'])
ax[4].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax[4].set_title('r='+str(np.round(r,2))+', PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[4].set_xlim(40,130)
ax[4].set_xticks((50,100))
ax[4].set_ylim(0,32)
ax[4].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('Ca vs. B9')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))
ax[4].text(0.65,0.91,'Mammalia',fontsize=fontSize,transform=ax[4].transAxes)

# circle the Cavia porcellus (guineaPig) and the other small tail species and put their name in text
additionalCircleSize = 7
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Cavia porcellus'])/1000+1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Cavia porcellus']+1.2,'$\it{Cavia}$\n$\it{porcellus}$',fontsize=fontSize-2,ha='left',va='bottom')
# now the others, which are Homo sapiens, Pan troglodytes, and Choloepus didactylus
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Homo sapiens'])/1000+2,intergenicAll['Ca'][intergenicAll[speciesKey]=='Homo sapiens']+0.5,'$\it{Homo}$\n$\it{sapiens}$',fontsize=fontSize-2,ha='left',va='bottom')
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Pan troglodytes'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pan troglodytes']+0.25,'$\it{Pan}$ $\it{troglodytes}$',fontsize=fontSize-2,ha='right',va='top')
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Choloepus didactylus'])/1000-3,intergenicAll['Ca'][intergenicAll[speciesKey]=='Choloepus didactylus'],'$\it{Choloepus}$\n$\it{didactylus}$',fontsize=fontSize-2,ha='right',va='bottom')
# and the mouse
ax[4].plot((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[4].text((intergenicAll['B9'][intergenicAll[speciesKey]=='Mus musculus'])/1000-4,intergenicAll['Ca'][intergenicAll[speciesKey]=='Mus musculus'],'$\it{Mus}$\n$\it{musculus}$',fontsize=fontSize-2,ha='right',va='top')

# plot Cervical vs. B4 for reptiles
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xKeptReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B4','C']),speciesKey,nameKey,classKey)
xNotKeptReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['C'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['C'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[5].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[5].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[5].text(0.05,0.05,'Reptilia',fontsize=fontSize,transform=ax[5].transAxes)
ax[5].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[5].set_ylabel('Cervical',fontsize=fontSize)
x = xKeptReptilia
y = yKeptReptilia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B4'],xPICn['C'])
# ax[5].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[5].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[5].set_xlim(10,30)
ax[5].set_ylim(0,10)
# set tick font size
ax[5].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('C vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a turtle
ax[5].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Chelonia mydas']+0.4,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[5].plot((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[5].text((1*intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['C'][intergenicAll[speciesKey]=='Zootoca vivipara']+0.4,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# plot Thoracic vs. the same intergenic distances
# no inset
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xKeptReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yKeptReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['B4','T']),speciesKey,nameKey,classKey)
xNotKeptReptilia = intergenicAll['B4'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
yNotKeptReptilia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['B4']))&(intergenicAll[classKey]=='Reptilia')]
markerSize = 6
ax[6].plot((xKeptReptilia)/1000,yKeptReptilia,marker='^',color=colorWheel[2],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[6].plot((xNotKeptReptilia)/1000,yNotKeptReptilia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[6].text(0.72,0.05,'Reptilia',fontsize=fontSize,transform=ax[6].transAxes)
ax[6].set_xlabel('intergenic distance\n$\it{B4}$-$\it{B5}$ (kb)',fontsize=fontSize)
ax[6].set_ylabel('Thoracic',fontsize=fontSize)
x = xKeptReptilia
y = yKeptReptilia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['B4'],xPICn['T'])
# ax[6].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
ax[6].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[6].set_xlim(10,30)
ax[6].set_ylim(5,30)
# set tick font size
ax[6].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. B4')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a turtle
ax[6].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[6].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Chelonia mydas'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Chelonia mydas']+1,'$\it{Chelonia}$\n$\it{mydas}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a lizard
ax[6].plot((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[6].text((intergenicAll['B4'][intergenicAll[speciesKey]=='Zootoca vivipara'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Zootoca vivipara']-1,'$\it{Zootoca}$\n$\it{vivipara}$',fontsize=fontSize-2,ha='right',va='bottom')

# Thoracic vs. D1 for amphibians
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
xKeptAmphibia = intergenicAll['D1'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
yKeptAmphibia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['D1','T']),speciesKey,nameKey,classKey)
xNotKeptAmphibia = intergenicAll['D1'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
yNotKeptAmphibia = intergenicAll['T'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['T'])&np.isfinite(intergenicAll['D1']))&(intergenicAll[classKey]=='Amphibia')]
markerSize = 6
ax[7].plot((xKeptAmphibia)/1000,yKeptAmphibia,marker='^',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[7].plot((xNotKeptAmphibia)/1000,yNotKeptAmphibia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)

ax[7].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[7].transAxes)
ax[7].set_xlabel('intergenic distance\n$\it{D1}$-$\it{D3}$ (kb)',fontsize=fontSize)
ax[7].set_ylabel('Thoracic',fontsize=fontSize)
x = xKeptAmphibia
y = yKeptAmphibia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['D1'],xPICn['T'])
# ax[7].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'**',fontsize=fontSize)
# ax[7].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[7].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a frog
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Xenopus tropicalis']+0.35,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-0.25,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[7].plot((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[7].text((intergenicAll['D1'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['T'][intergenicAll[speciesKey]=='Pleurodeles waltl']-0.25,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# and finally amphibian Ca vs. A11
speciesKept = intergenicAll['Species'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
xKeptAmphibia = intergenicAll['A11'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
yKeptAmphibia = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='yes')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
xPICn,xPIC = picFunction(tree,intergenicAll,speciesKept,(['A11','Ca']),speciesKey,nameKey,classKey)
xNotKeptAmphibia = intergenicAll['A11'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
yNotKeptAmphibia = intergenicAll['Ca'][(intergenicAll[nameKey]!='snake')&(intergenicAll['Kept']=='no')&(np.isfinite(intergenicAll['Ca'])&np.isfinite(intergenicAll['A11']))&(intergenicAll[classKey]=='Amphibia')]
markerSize = 6
ax[8].plot((xKeptAmphibia)/1000,yKeptAmphibia,marker='^',color=colorWheel[3],linestyle='None',alpha=0.5,markersize=markerSize,label='$\mathregular{\mathcal{R}}$')
# plot the not annotated as empty symbols
ax[8].plot((xNotKeptAmphibia)/1000,yNotKeptAmphibia,marker='^',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize)
x = xKeptAmphibia
y = yKeptAmphibia
r,p = scipy.stats.pearsonr(x[(~np.isnan(x))&(~np.isnan(y))],y[(~np.isnan(x))&(~np.isnan(y))])
rPIC,pPIC = scipy.stats.pearsonr(xPICn['A11'],xPICn['Ca'])
ax[8].text(0.05,0.92,'Amphibia',fontsize=fontSize,transform=ax[8].transAxes)
ax[8].set_xlabel('intergenic distance\n$\it{A11}$-$\it{A12}$ (kb)',fontsize=fontSize)
ax[8].set_ylabel('Caudal',fontsize=fontSize)
# ax[8].set_title('r='+str(np.round(r,2))+'***, PIC: r='+str(np.round(rPIC,2))+'',fontsize=fontSize)
# ax[8].set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax[8].set_xlim(0,70)
# set tick font size
ax[8].tick_params(axis='both', which='major', labelsize=fontSize)
# print the r and p values
print('T vs. A2+D1')
print('r='+str(np.round(r,2))+', p='+"{:.2e}".format(p))
print('PIC: r='+str(np.round(rPIC,2))+', p='+"{:.2e}".format(pPIC))

# mark a frog
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Xenopus tropicalis'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Xenopus tropicalis']+1,'$\it{Xenopus}$\n$\it{tropicalis}$',fontsize=fontSize-2,ha='left',va='bottom')
# mark a axolotl
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Ambystoma mexicanum'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Ambystoma mexicanum']-1,'$\it{Ambystoma}$\n$\it{mexicanum}$',fontsize=fontSize-2,ha='right',va='top')
# mark a newt
ax[8].plot((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax[8].text((intergenicAll['A11'][intergenicAll[speciesKey]=='Pleurodeles waltl'])/1000-1,intergenicAll['Ca'][intergenicAll[speciesKey]=='Pleurodeles waltl']-1,'$\it{Pleurodeles}$\n$\it{waltl}$',fontsize=fontSize-2,ha='right',va='top')

# subplot labels
ax[0].text(-0.2,1.05,'A',fontsize=fontSize+4,fontweight='normal',transform=ax[0].transAxes)
ax[1].text(-0.2,1.05,'B',fontsize=fontSize+4,fontweight='normal',transform=ax[1].transAxes)
ax[2].text(-0.2,1.05,'C',fontsize=fontSize+4,fontweight='normal',transform=ax[2].transAxes)
ax[3].text(-0.2,1.05,'D',fontsize=fontSize+4,fontweight='normal',transform=ax[3].transAxes)
ax[4].text(-0.2,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax[4].transAxes)
ax[5].text(-0.2,1.05,'F',fontsize=fontSize+4,fontweight='normal',transform=ax[5].transAxes)
ax[6].text(-0.2,1.05,'G',fontsize=fontSize+4,fontweight='normal',transform=ax[6].transAxes)
ax[7].text(-0.2,1.05,'H',fontsize=fontSize+4,fontweight='normal',transform=ax[7].transAxes)
ax[8].text(-0.2,1.05,'I',fontsize=fontSize+4,fontweight='normal',transform=ax[8].transAxes)

# save
plt.savefig(outputPath+'plots/intergenicFiltered_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/intergenicFiltered_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')
