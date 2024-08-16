# combine all the vertebral results and pic analysis
# plot (Fig. 2 in the paper)
# plot Extended Data Figs. 1, 4, 5
# v2
# also outputs the data from the main figures for testing the PIC correlations with the R library "ape" separately

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath
outputPath = inputPath

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
import os

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
colorMap = 'coolwarm' # or 'gray' or 'bwr' or 'RdBu'
colorMapVariance = 'Reds' #
import matplotlib.gridspec as gridspec

import scipy.stats
markerWheel = ['o','s','^','>','<','v','d'] # pour convenience
fontSize = 12
faceColor = 'white'
markerSize = 0.5
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

thresholdConstraintPIC = 0.2 # except make an exception for the C+T archosaur constraint
statistic = 'pearson' # or 'spearman'

#%% check if there is a plots directory on the outputPath, and if not, make one

savePath = outputPath + 'plots/'
if not os.path.exists(savePath):
    os.makedirs(savePath)

#%% Define a function to check if one node is a descendant of another (from ChatGPT)
def is_descendant(tree, potential_descendant, target):
    # Start from the potential descendant node
    current_node = tree.find(potential_descendant)

    # Traverse the tree towards the root
    while current_node is not None:
        if current_node.name == target:
            return True  # Found the target node
        current_node = current_node.parent

    return False  # Target node not found along the path

#%% list the nodes in the tree and the number of their children

def listNodes(tree):
    nodes = []
    for node in tree.traverse():
        nodes.append(node.name)
    # get the number of children for each node
    # this is the number of tips that are descendants of that node
    numChildren = []
    for i in range(1,len(nodes)):
        cnt = 0
        for tips in tree.find(nodes[i]).tips():
            cnt = cnt + 1
        numChildren.append(cnt)
    return nodes, numChildren

#%% a function for saving the two traits that should be compared along with their pruned tree

def saveForPIC(speciesList,feature0,feature1,database,treeToSave,suffix):

    # dataframe for saving
    saveData = pd.DataFrame(columns=['species',feature0,feature1])
    # check if the 'treeSpecies' column is in the database, otherwise make it
    if 'treeSpecies' not in database.columns:
        database['treeSpecies'] = [database['species'].iloc[i].replace(' ','_') for i in range(len(database))]
    saveData['treeSpecies'] = database['treeSpecies'][database['species'].isin(speciesList)]
    saveData[feature0] = database[feature0][database['species'].isin(speciesList)]
    saveData[feature1] = database[feature1][database['species'].isin(speciesList)]
    # save this data for double-checking the PIC with other software
    saveData.to_csv(outputPath+'vertebral/vertebralData_'+feature0+'_'+feature1+suffix+'.csv',index=False)
    # get a reduced version of the tree corresponding to these species
    treeReduced = treeToSave.copy()
    # prune the tree to only include the species in the vertebralData and save
    treeReduced = treeReduced.shear(speciesList)
    treeReduced.prune()
    treeReduced.write(outputPath+'vertebral/vertebralData_'+feature0+'_'+feature1+suffix+'_tree.nwk',format='newick')

#%% function to get the number of asterisks to use for the p-value

def getAsterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

#%% load the tree

treePath = outputPath+'vertebral/fullFormulaTree/tree.nwk'
with open(treePath) as f:
    treeFile = f.read()
tree = TreeNode.read(StringIO(treeFile))
    
#%% load the species list and the features

# load the only .csv file in the inputPath, so search for it with glob
dataPath = glob.glob(inputPath+'*vertebralFormulaOrdered_v2.csv')[0]
vertebral = pd.read_csv(dataPath)

# need to rename some columns
vertebral = vertebral.rename(columns={'Cervical':'C','Thoracic':'T','Lumbar':'L','Sacral':'S','Caudal':'Ca','Class':'class','Species':'species'})
classKey = 'class'

# add a 'name' column
commonName = []
for i in range(len(vertebral)):
    tempName = vertebral['Common name'].iloc[i]
    # remove all spaces and capitalize the first letter of each word
    tempNameSplit = tempName.split(' ')
    tempNameSplit = [tempNameSplit[i].capitalize() for i in range(len(tempNameSplit))]
    tempName = ''.join(tempNameSplit)
    commonName.append(tempName)
nameKey = 'name'
vertebral.insert(0,nameKey,commonName)

dataCols = ['C','T','L','S','Ca']

#%% PIC data from "picVertebral.py"

vertebralPIC = pd.read_csv(outputPath+'vertebral/pic_normalized_vertebral.csv')
vertebralPIC_raw = pd.read_csv(outputPath+'vertebral/pic_raw_vertebral.csv')
PIC = vertebralPIC.copy()
PICraw = vertebralPIC_raw.copy()

#%% load constraints and plasticities from "getVertebral_postAnalysis.py"

constraints = pd.read_csv(outputPath+'vertebral/constraintMasterUnique_organized.csv')
plasticities = pd.read_csv(outputPath+'vertebral/plasticityMasterUnique_organized.csv')
constraintsAll = pd.read_csv(outputPath+'vertebral/constraintMaster.csv')
plasticitiesAll = pd.read_csv(outputPath+'vertebral/plasticityMaster.csv')

#%% make a list of each of the species lists for each PIC datapoint

speciesList = []
for i in range(len(vertebralPIC)):
    temp = vertebralPIC['species'].iloc[i]
    # replace all '(' and ')' with nothing
    temp = temp.replace('(','')
    temp = temp.replace(')','')
    # split by + and -
    temp = temp.split('+')
    temp = [temp[i].split('-') for i in range(len(temp))]
    # flatten the list
    temp = [item for sublist in temp for item in sublist]
    speciesList.append(temp)

#%% loop through the constraints and plasticities, only testing the correlation on the species in that node
# do this for both the original data and the data after PIC
# we can't do this for the constraints or plasticities which are simply one vertebral count (e.g. just cervical or just caudal)

constraintR = np.zeros((len(constraints),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
constraintP = np.zeros((len(constraints),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
plasticityR = np.zeros((len(constraints),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
plasticityP = np.zeros((len(constraints),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC

prefactor = np.zeros(5)
prefactorSave = np.zeros((len(constraints),5)) # save the prefactors for each constraint
leftSave = []
rightSave = []
indSingle = np.zeros(len(constraints)) # save the indices of the constraints that are single (e.g. only cervical or only caudal
indCT = np.zeros(len(constraints)) # save the indices of the constraints that are C+T

# constraints (unique)
for i in range(len(constraints)):
    node = constraints['node'].iloc[i]
    prefactor[0] = constraints['C'].iloc[i]
    prefactor[1] = constraints['T'].iloc[i]
    prefactor[2] = constraints['L'].iloc[i]
    prefactor[3] = constraints['S'].iloc[i]
    prefactor[4] = constraints['Ca'].iloc[i]
    # save the index of the one C+T constraint found
    if (node == '1351') & (constraints['C'].iloc[i]==1) & (constraints['T'].iloc[i]==1):
        indCT[i] = 1
    # prefactorSave[i,:] = prefactor
    if np.count_nonzero(prefactor) > 1: # check if more than one nonzero prefactor for C,T,L,S,Ca
        # split up the nonzero prefactors for the correlation
        prefactorNonZero = prefactor[prefactor!=0]
        indNonZero = np.where(prefactor!=0)[0]
        # check if any are negative, in that case let the "left" be the positive and the "right" be the negative
        left = np.zeros(5)
        right = np.zeros(5)
        if np.any(prefactorNonZero<0):
            left[prefactor>0] = prefactor[prefactor>0]
            right[prefactor<0] = prefactor[prefactor<0]
            right = -right
        else:
            if len(prefactorNonZero) == 2: # only two nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                right[indNonZero[1]] = prefactorNonZero[1]
            elif len(prefactorNonZero) == 3: # three nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                right[indNonZero[2]] = prefactorNonZero[2]
            elif len(prefactorNonZero) == 4: # four nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                right[indNonZero[2]] = prefactorNonZero[2]
                right[indNonZero[3]] = prefactorNonZero[3]
            elif len(prefactorNonZero) == 5: # five nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                left[indNonZero[2]] = prefactorNonZero[2]
                right[indNonZero[3]] = prefactorNonZero[3]
                right[indNonZero[4]] = prefactorNonZero[4]
        leftSave.append(left)
        rightSave.append(right)
        # now check the nodes
        if node == 'full': # full tree
            if statistic == 'pearson':
                r,p = scipy.stats.pearsonr(vertebral.loc[:,dataCols].dot(left),vertebral.loc[:,dataCols].dot(right))
                rPIC,pPIC = scipy.stats.pearsonr(vertebralPIC.loc[:,dataCols].dot(left),vertebralPIC.loc[:,dataCols].dot(right))
                rPIC_raw,pPIC_raw = scipy.stats.pearsonr(vertebralPIC_raw.loc[:,dataCols].dot(left),vertebralPIC_raw.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                r,p = scipy.stats.spearmanr(vertebral.loc[:,dataCols].dot(left),vertebral.loc[:,dataCols].dot(right))
                rPIC,pPIC = scipy.stats.spearmanr(vertebralPIC.loc[:,dataCols].dot(left),vertebralPIC.loc[:,dataCols].dot(right))
                rPIC_raw,pPIC_raw = scipy.stats.spearmanr(vertebralPIC_raw.loc[:,dataCols].dot(left),vertebralPIC_raw.loc[:,dataCols].dot(right))
        else:
            # need to get the species in this node
            # find the directory called "subTree_node_..." in the vertebral data
            subtreeDir = glob.glob(outputPath+'vertebral/subTree_'+node+'*')[0]
            species = pd.read_csv(subtreeDir+'/species.csv',header=None)[0].to_list()
            vertebralTemp = vertebral[vertebral['species'].isin(species)]
            if statistic == 'pearson':
                r,p = scipy.stats.pearsonr(vertebralTemp.loc[:,dataCols].dot(left),vertebralTemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                r,p = scipy.stats.spearmanr(vertebralTemp.loc[:,dataCols].dot(left),vertebralTemp.loc[:,dataCols].dot(right))
            # find the PIC datapoints that have all of these species or differ by only one
            # we will use the speciesList to do this
            # first, find the indices of the PIC data that have all of these species
            # we will do this by finding the intersection of the species list and the species list of the node
            # we will do this by converting the species list to a set
            speciesSet = set(species)
            # then we will loop through the species list and find the intersection of the set and the species list
            # we will do this for each PIC datapoint
            # we will also keep track of the indices of the PIC datapoints that have all of these species
            indices = []
            for j in range(len(speciesList)):
                # if speciesSet.intersection(speciesList[j]) == speciesSet:
                if set(speciesList[j]).intersection(speciesSet) == set(speciesList[j]):
                    indices.append(j)
            vertebralPICtemp = vertebralPIC.iloc[indices]
            if statistic == 'pearson':
                rPIC,pPIC = scipy.stats.pearsonr(vertebralPICtemp.loc[:,dataCols].dot(left),vertebralPICtemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                rPIC,pPIC = scipy.stats.spearmanr(vertebralPICtemp.loc[:,dataCols].dot(left),vertebralPICtemp.loc[:,dataCols].dot(right))
            vertebralPIC_rawTemp = vertebralPIC_raw.iloc[indices]
            if statistic == 'pearson':
                rPIC_raw,pPIC_raw = scipy.stats.pearsonr(vertebralPIC_rawTemp.loc[:,dataCols].dot(left),vertebralPIC_rawTemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                rPIC_raw,pPIC_raw = scipy.stats.spearmanr(vertebralPIC_rawTemp.loc[:,dataCols].dot(left),vertebralPIC_rawTemp.loc[:,dataCols].dot(right))
        constraintR[i,0] = r
        constraintP[i,0] = p
        constraintR[i,1] = rPIC
        constraintP[i,1] = pPIC
        constraintR[i,2] = rPIC_raw
        constraintP[i,2] = pPIC_raw
    else:
        indSingle[i] = 1
        constraintR[i,0] = np.nan
        constraintP[i,0] = np.nan
        constraintR[i,1] = np.nan
        constraintP[i,1] = np.nan
        constraintR[i,2] = np.nan
        constraintP[i,2] = np.nan
        leftSave.append(np.array([0,0,0,0,0]))
        rightSave.append(np.array([0,0,0,0,0]))
        
# reduce the constraint data to only the ones above a threshold or with only one prefactor (so leftSave = np.nan)
indSave = np.where((np.abs(constraintR[:,1])>thresholdConstraintPIC)|(indSingle==1)|(indCT==1))[0]
constraintsReduced = constraints.iloc[indSave]
# reset the index
constraintsReduced = constraintsReduced.reset_index(drop=True)
# add the new constraint R and P values
constraintsReduced['r'] = constraintR[indSave,0]
constraintsReduced['p'] = constraintP[indSave,0]
constraintsReduced['rPIC'] = constraintR[indSave,1]
constraintsReduced['pPIC'] = constraintP[indSave,1]
constraintsReduced['rPIC_raw'] = constraintR[indSave,2]
constraintsReduced['pPIC_raw'] = constraintP[indSave,2]
# add the left and right prefactors
constraintsReduced['left'] = list(np.array(leftSave)[indSave].astype(int))
constraintsReduced['right'] = list(np.array(rightSave)[indSave].astype(int))

# constraints (all)
constraintRall = np.zeros((len(constraintsAll),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
constraintPall = np.zeros((len(constraintsAll),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
plasticityRall = np.zeros((len(constraintsAll),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC
plasticityPall = np.zeros((len(constraintsAll),3)) # first is before PIC, second is after with normalized PIC, then with raw PIC

prefactor = np.zeros(5)
prefactorSaveAll = np.zeros((len(constraintsAll),5)) # save the prefactors for each constraint
leftSaveAll = []
rightSaveAll = []
indSingleAll = np.zeros(len(constraintsAll)) # save the indices of the constraints that are single (e.g. only cervical or only caudal
indCTAll = np.zeros(len(constraintsAll)) # save the indices of the constraints that are C+T

for i in range(len(constraintsAll)):
    node = constraintsAll['node'].iloc[i]
    prefactor[0] = constraintsAll['C'].iloc[i]
    prefactor[1] = constraintsAll['T'].iloc[i]
    prefactor[2] = constraintsAll['L'].iloc[i]
    prefactor[3] = constraintsAll['S'].iloc[i]
    prefactor[4] = constraintsAll['Ca'].iloc[i]
    # prefactorSave[i,:] = prefactor
    if (node == '1351') & (constraintsAll['C'].iloc[i]==1) & (constraintsAll['T'].iloc[i]==1):
        indCTAll[i] = 1
    if np.count_nonzero(prefactor) > 1: # check if more than one nonzero prefactor for C,T,L,S,Ca
        # split up the nonzero prefactors for the correlation
        prefactorNonZero = prefactor[prefactor!=0]
        indNonZero = np.where(prefactor!=0)[0]
        # check if any are negative, in that case let the "left" be the positive and the "right" be the negative
        left = np.zeros(5)
        right = np.zeros(5)
        if np.any(prefactorNonZero<0):
            left[prefactor>0] = prefactor[prefactor>0]
            right[prefactor<0] = prefactor[prefactor<0]
            right = -right
        else:
            if len(prefactorNonZero) == 2: # only two nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                right[indNonZero[1]] = prefactorNonZero[1]
            elif len(prefactorNonZero) == 3: # three nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                right[indNonZero[2]] = prefactorNonZero[2]
            elif len(prefactorNonZero) == 4: # four nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                right[indNonZero[2]] = prefactorNonZero[2]
                right[indNonZero[3]] = prefactorNonZero[3]
            elif len(prefactorNonZero) == 5: # five nonzero prefactors
                left[indNonZero[0]] = prefactorNonZero[0]
                left[indNonZero[1]] = prefactorNonZero[1]
                left[indNonZero[2]] = prefactorNonZero[2]
                right[indNonZero[3]] = prefactorNonZero[3]
                right[indNonZero[4]] = prefactorNonZero[4]
        leftSaveAll.append(left)
        rightSaveAll.append(right)
        # now check the nodes
        if node == 'full': # full tree
            if statistic == 'pearson':
                r,p = scipy.stats.pearsonr(vertebral.loc[:,dataCols].dot(left),vertebral.loc[:,dataCols].dot(right))
                rPIC,pPIC = scipy.stats.pearsonr(vertebralPIC.loc[:,dataCols].dot(left),vertebralPIC.loc[:,dataCols].dot(right))
                rPIC_raw,pPIC_raw = scipy.stats.pearsonr(vertebralPIC_raw.loc[:,dataCols].dot(left),vertebralPIC_raw.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                r,p = scipy.stats.spearmanr(vertebral.loc[:,dataCols].dot(left),vertebral.loc[:,dataCols].dot(right))
                rPIC,pPIC = scipy.stats.spearmanr(vertebralPIC.loc[:,dataCols].dot(left),vertebralPIC.loc[:,dataCols].dot(right))
                rPIC_raw,pPIC_raw = scipy.stats.spearmanr(vertebralPIC_raw.loc[:,dataCols].dot(left),vertebralPIC_raw.loc[:,dataCols].dot(right))
        else:
            # need to get the species in this node
            # find the directory called "subTree_node_..." in the vertebral data
            subtreeDir = glob.glob(outputPath+'vertebral/subTree_'+node+'*')[0]
            species = pd.read_csv(subtreeDir+'/species.csv',header=None)[0].to_list()
            vertebralTemp = vertebral[vertebral['species'].isin(species)]
            if statistic == 'pearson':
                r,p = scipy.stats.pearsonr(vertebralTemp.loc[:,dataCols].dot(left),vertebralTemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                r,p = scipy.stats.spearmanr(vertebralTemp.loc[:,dataCols].dot(left),vertebralTemp.loc[:,dataCols].dot(right))
            # find the PIC datapoints that have all of these species or differ by only one
            # we will use the speciesList to do this
            # first, find the indices of the PIC data that have all of these species
            # we will do this by finding the intersection of the species list and the species list of the node
            # we will do this by converting the species list to a set
            speciesSet = set(species)
            # then we will loop through the species list and find the intersection of the set and the species list
            # we will do this for each PIC datapoint
            # we will also keep track of the indices of the PIC datapoints that have all of these species
            indices = []
            for j in range(len(speciesList)):
                # if speciesSet.intersection(speciesList[j]) == speciesSet:
                if set(speciesList[j]).intersection(speciesSet) == set(speciesList[j]):
                    indices.append(j)
            vertebralPICtemp = vertebralPIC.iloc[indices]
            if statistic == 'pearson':
                rPIC,pPIC = scipy.stats.pearsonr(vertebralPICtemp.loc[:,dataCols].dot(left),vertebralPICtemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                rPIC,pPIC = scipy.stats.spearmanr(vertebralPICtemp.loc[:,dataCols].dot(left),vertebralPICtemp.loc[:,dataCols].dot(right))
            vertebralPIC_rawTemp = vertebralPIC_raw.iloc[indices]
            if statistic == 'pearson':
                rPIC_raw,pPIC_raw = scipy.stats.pearsonr(vertebralPIC_rawTemp.loc[:,dataCols].dot(left),vertebralPIC_rawTemp.loc[:,dataCols].dot(right))
            elif statistic == 'spearman':
                rPIC_raw,pPIC_raw = scipy.stats.spearmanr(vertebralPIC_rawTemp.loc[:,dataCols].dot(left),vertebralPIC_rawTemp.loc[:,dataCols].dot(right))
        constraintRall[i,0] = r
        constraintPall[i,0] = p
        constraintRall[i,1] = rPIC
        constraintPall[i,1] = pPIC
        constraintRall[i,2] = rPIC_raw
        constraintPall[i,2] = pPIC_raw
    else:
        indSingleAll[i] = 1
        constraintRall[i,0] = np.nan
        constraintPall[i,0] = np.nan
        constraintRall[i,1] = np.nan
        constraintPall[i,1] = np.nan
        constraintRall[i,2] = np.nan
        constraintPall[i,2] = np.nan
        leftSaveAll.append(np.array([0,0,0,0,0]))
        rightSaveAll.append(np.array([0,0,0,0,0]))
        
# reduce the constraint data to only the ones above a threshold or with only one prefactor (so leftSave = np.nan)
indSaveAll = np.where((np.abs(constraintRall[:,1])>thresholdConstraintPIC)|(indSingleAll==1)|(indCTAll==1))[0]
constraintsAllReduced = constraintsAll.iloc[indSaveAll]
# reset the index
constraintsAllReduced = constraintsAllReduced.reset_index(drop=True)
# add the new constraint R and P values
constraintsAllReduced['r'] = constraintRall[indSaveAll,0]
constraintsAllReduced['p'] = constraintPall[indSaveAll,0]
constraintsAllReduced['rPIC'] = constraintRall[indSaveAll,1]
constraintsAllReduced['pPIC'] = constraintPall[indSaveAll,1]
constraintsAllReduced['rPIC_raw'] = constraintRall[indSaveAll,2]
constraintsAllReduced['pPIC_raw'] = constraintPall[indSaveAll,2]
# add the left and right prefactors
constraintsAllReduced['left'] = list(np.array(leftSaveAll)[indSaveAll].astype(int))
constraintsAllReduced['right'] = list(np.array(rightSaveAll)[indSaveAll].astype(int))

#%% now go through each unique constraint and see if a constraint is shared by multiple nodes
# I defined a function to check if one node is the descendant of another

# first make a list of all the unique constraints, which are the combinations of C,T,L,S,Ca
constraintsAllReducedList = []
for i in range(len(constraintsAllReduced)):
    constraintsAllReducedList.append(tuple(constraintsAllReduced.iloc[i,1:6]))
constraintsUnique = list(set(constraintsAllReducedList))

# for all the unique constraints, get the index of the constraintsAll that has that constraint
indConstraint = np.zeros(len(constraintsAllReducedList))
for i in range(len(constraintsUnique)):
    # check that all 5 columns are the same
    ind = np.where(np.all(constraintsAllReduced.iloc[:,1:6]==constraintsUnique[i],axis=1))[0]
    indConstraint[ind] = i
    
# get the list of nodes for each unique constraint
nodesUnique = []
insideVarianceUnique = []
totalVarianceUnique = []
explainedVarianceUnique = []
outsideVarianceUnique = []
numMammalsUnique = []
numBirdsUnique = []
numReptilesUnique = []
numAmphibiansUnique = []
rUnique = []
pUnique = []
rPICUnique = []
pPICUnique = []
rPIC_rawUnique = []
pPIC_rawUnique = []
leftUnique = []
rightUnique = []
for i in range(len(constraintsUnique)):
    tempNodeList = constraintsAllReduced['node'].iloc[np.where(indConstraint==i)[0]].to_list()
    # get only the unique nodes, and we need the indices for the rest of the data
    _, ind = np.unique(tempNodeList,return_index=True)
    nodesUnique.append(np.array(constraintsAllReduced['node'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    insideVarianceUnique.append(np.array(constraintsAllReduced['insideVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    totalVarianceUnique.append(np.array(constraintsAllReduced['totalVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    explainedVarianceUnique.append(np.array(constraintsAllReduced['explainedVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    outsideVarianceUnique.append(np.array(constraintsAllReduced['outsideVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numMammalsUnique.append(np.array(constraintsAllReduced['numMammals'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numBirdsUnique.append(np.array(constraintsAllReduced['numBirds'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numReptilesUnique.append(np.array(constraintsAllReduced['numReptiles'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numAmphibiansUnique.append(np.array(constraintsAllReduced['numAmphibians'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    rUnique.append(np.array(constraintsAllReduced['r'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    pUnique.append(np.array(constraintsAllReduced['p'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    rPICUnique.append(np.array(constraintsAllReduced['rPIC'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    pPICUnique.append(np.array(constraintsAllReduced['pPIC'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    rPIC_rawUnique.append(np.array(constraintsAllReduced['rPIC_raw'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    pPIC_rawUnique.append(np.array(constraintsAllReduced['pPIC_raw'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    leftUnique.append(np.array(constraintsAllReduced['left'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    rightUnique.append(np.array(constraintsAllReduced['right'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])

# now iterate through each unique constraint, and then iterate through all the nodes
# keep only the nodes which have no ancestor node in the list of nodes for that constraint
# (this will require 3 nested loops I think)
nodesUniqueKeep = []
insideVarianceKeep = []
totalVarianceKeep = []
explainedVarianceKeep = []
outsideVarianceKeep = []
numMammalsKeep = []
numBirdsKeep = []
numReptilesKeep = []
numAmphibiansKeep = []
rUniqueKeep = []
pUniqueKeep = []
rPICUniqueKeep = []
pPICUniqueKeep = []
rPIC_rawUniqueKeep = []
pPIC_rawUniqueKeep = []
leftUniqueKeep = []
rightUniqueKeep = []
for i in range(len(constraintsUnique)):
    nodesUniqueKeepTemp = []
    insideVarianceKeepTemp = []
    totalVarianceKeepTemp = []
    explainedVarianceKeepTemp = []
    outsideVarianceKeepTemp = []
    numMammalsKeepTemp = []
    numBirdsKeepTemp = []
    numReptilesKeepTemp = []
    numAmphibiansKeepTemp = []
    rUniqueKeepTemp = []
    pUniqueKeepTemp = []
    rPICUniqueKeepTemp = []
    pPICUniqueKeepTemp = []
    rPIC_rawUniqueKeepTemp = []
    pPIC_rawUniqueKeepTemp = []
    leftUniqueKeepTemp = []
    rightUniqueKeepTemp = []
    # if 'full' is in the list of nodes, we know it is the ancestor of all others, so we can skip using the is_descendant function
    if 'full' in nodesUnique[i]:
        indFull = np.where(np.array(nodesUnique[i])=='full')[0][0]
        nodesUniqueKeepTemp.append(nodesUnique[i][indFull])
        insideVarianceKeepTemp.append(insideVarianceUnique[i][indFull])
        totalVarianceKeepTemp.append(totalVarianceUnique[i][indFull])
        explainedVarianceKeepTemp.append(explainedVarianceUnique[i][indFull])
        outsideVarianceKeepTemp.append(outsideVarianceUnique[i][indFull])
        numMammalsKeepTemp.append(numMammalsUnique[i][indFull])
        numBirdsKeepTemp.append(numBirdsUnique[i][indFull])
        numReptilesKeepTemp.append(numReptilesUnique[i][indFull])
        numAmphibiansKeepTemp.append(numAmphibiansUnique[i][indFull])
        rUniqueKeepTemp.append(rUnique[i][indFull])
        pUniqueKeepTemp.append(pUnique[i][indFull])
        rPICUniqueKeepTemp.append(rPICUnique[i][indFull])
        pPICUniqueKeepTemp.append(pPICUnique[i][indFull])
        rPIC_rawUniqueKeepTemp.append(rPIC_rawUnique[i][indFull])
        pPIC_rawUniqueKeepTemp.append(pPIC_rawUnique[i][indFull])
        leftUniqueKeepTemp.append(leftUnique[i][indFull])
        rightUniqueKeepTemp.append(rightUnique[i][indFull])
    else:
        for j in range(len(nodesUnique[i])):
            keep = True
            for k in range(len(nodesUnique[i])):
                if k != j:
                    if is_descendant(tree,potential_descendant=nodesUnique[i][j],target=nodesUnique[i][k]):
                        keep = False
            if keep:
                nodesUniqueKeepTemp.append(nodesUnique[i][j])
                insideVarianceKeepTemp.append(insideVarianceUnique[i][j])
                totalVarianceKeepTemp.append(totalVarianceUnique[i][j])
                explainedVarianceKeepTemp.append(explainedVarianceUnique[i][j])
                outsideVarianceKeepTemp.append(outsideVarianceUnique[i][j])
                numMammalsKeepTemp.append(numMammalsUnique[i][j])
                numBirdsKeepTemp.append(numBirdsUnique[i][j])
                numReptilesKeepTemp.append(numReptilesUnique[i][j])
                numAmphibiansKeepTemp.append(numAmphibiansUnique[i][j])
                rUniqueKeepTemp.append(rUnique[i][j])
                pUniqueKeepTemp.append(pUnique[i][j])
                rPICUniqueKeepTemp.append(rPICUnique[i][j])
                pPICUniqueKeepTemp.append(pPICUnique[i][j])
                rPIC_rawUniqueKeepTemp.append(rPIC_rawUnique[i][j])
                pPIC_rawUniqueKeepTemp.append(pPIC_rawUnique[i][j])
                leftUniqueKeepTemp.append(leftUnique[i][j])
                rightUniqueKeepTemp.append(rightUnique[i][j])
    nodesUniqueKeep.append(nodesUniqueKeepTemp)
    insideVarianceKeep.append(insideVarianceKeepTemp)
    totalVarianceKeep.append(totalVarianceKeepTemp)
    explainedVarianceKeep.append(explainedVarianceKeepTemp)
    outsideVarianceKeep.append(outsideVarianceKeepTemp)
    numMammalsKeep.append(numMammalsKeepTemp)
    numBirdsKeep.append(numBirdsKeepTemp)
    numReptilesKeep.append(numReptilesKeepTemp)
    numAmphibiansKeep.append(numAmphibiansKeepTemp)
    rUniqueKeep.append(rUniqueKeepTemp)
    pUniqueKeep.append(pUniqueKeepTemp)
    rPICUniqueKeep.append(rPICUniqueKeepTemp)
    pPICUniqueKeep.append(pPICUniqueKeepTemp)
    rPIC_rawUniqueKeep.append(rPIC_rawUniqueKeepTemp)
    pPIC_rawUniqueKeep.append(pPIC_rawUniqueKeepTemp)
    leftUniqueKeep.append(leftUniqueKeepTemp)
    rightUniqueKeep.append(rightUniqueKeepTemp)
    
# now make a dataframe with the nodesUniqueKeep and the constraintsUnique
nodeList = []
cervicalList = []
thoracicList = []
lumbarList = []
sacralList = []
caudalList = []
insideVarianceList = []
totalVarianceList = []
explainedVarianceList = []
outsideVarianceList = []
numMammalsList = []
numBirdsList = []
numReptilesList = []
numAmphibiansList = []
rUniqueList = []
pUniqueList = []
rPICUniqueList = []
pPICUniqueList = []
rPIC_rawUniqueList = []
pPIC_rawUniqueList = []
leftUniqueList = []
rightUniqueList = []
for i in range(len(constraintsUnique)):
    for j in range(len(nodesUniqueKeep[i])):
        nodeList.append(nodesUniqueKeep[i][j])
        cervicalList.append(constraintsUnique[i][0])
        thoracicList.append(constraintsUnique[i][1])
        lumbarList.append(constraintsUnique[i][2])
        sacralList.append(constraintsUnique[i][3])
        caudalList.append(constraintsUnique[i][4])
        insideVarianceList.append(insideVarianceKeep[i][j])
        totalVarianceList.append(totalVarianceKeep[i][j])
        explainedVarianceList.append(explainedVarianceKeep[i][j])
        outsideVarianceList.append(outsideVarianceKeep[i][j])
        numMammalsList.append(numMammalsKeep[i][j])
        numBirdsList.append(numBirdsKeep[i][j])
        numReptilesList.append(numReptilesKeep[i][j])
        numAmphibiansList.append(numAmphibiansKeep[i][j])
        rUniqueList.append(rUniqueKeep[i][j])
        pUniqueList.append(pUniqueKeep[i][j])
        rPICUniqueList.append(rPICUniqueKeep[i][j])
        pPICUniqueList.append(pPICUniqueKeep[i][j])
        rPIC_rawUniqueList.append(rPIC_rawUniqueKeep[i][j])
        pPIC_rawUniqueList.append(pPIC_rawUniqueKeep[i][j])
        leftUniqueList.append(leftUniqueKeep[i][j])
        rightUniqueList.append(rightUniqueKeep[i][j])
        
constraintsAllReducedUnique = pd.DataFrame({'node':nodeList,'C':cervicalList,'T':thoracicList,'L':lumbarList,'S':sacralList,'Ca':caudalList,'insideVariance':insideVarianceList,'totalVariance':totalVarianceList,'explainedVariance':explainedVarianceList,'outsideVariance':outsideVarianceList,'numMammals':numMammalsList,'numBirds':numBirdsList,'numReptiles':numReptilesList,'numAmphibians':numAmphibiansList,'r':rUniqueList,'p':pUniqueList,'rPIC':rPICUniqueList,'pPIC':pPICUniqueList,'rPIC_raw':rPIC_rawUniqueList,'pPIC_raw':pPIC_rawUniqueList,'left':leftUniqueList,'right':rightUniqueList})

#%% find the difference between the two pandas dataframes constraintsAllReducedUnique and constraintsReduced
# just compare the first 6 columns
# it seems that the bird one is the only one that had this issue!

dfDiff = pd.concat([constraintsAllReducedUnique.iloc[:,:6],constraintsReduced.iloc[:,:6]]).drop_duplicates(keep=False)

#%% get a sorted version of the constraintsAllReducedUnique dataframe by node

constraintsSortedByNode = constraintsAllReducedUnique.sort_values(by=['node'],ascending=False)

#%% do hierarchical clustering on the constraints (the formula C,T,L,S,Ca)

postSort = False

# first, we need to make a matrix of the constraints
def corrMatrixFunction(constraints,startCol,endCol):
    # the pandas native way gives nans for some reason...
    corrMat = np.zeros((len(constraints),len(constraints)))
    for i in range(len(constraints)):
        for j in range(len(constraints)):
            corrMat[i,j] = np.inner(constraints.iloc[i,startCol:endCol].values,constraints.iloc[j,startCol:endCol].values)/(np.sqrt(np.inner(constraints.iloc[i,startCol:endCol].values,constraints.iloc[i,startCol:endCol].values))*np.sqrt(np.inner(constraints.iloc[j,startCol:endCol].values,constraints.iloc[j,startCol:endCol].values)))
    return corrMat

methodName = 'complete'
# methodName = 'single'
# methodName = 'centroid'

corrMat = corrMatrixFunction(constraintsAllReducedUnique,1,6)

# the pandas native way gives nans for some reason...
pdist = spc.distance.pdist(corrMat)
linkage = spc.linkage(pdist, method=methodName)
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
# add the cluster to the dataframe
constraintsAllReducedUnique['cluster'] = idx
# determine which cluster has the most constituents
unique, counts = np.unique(idx, return_counts=True)
newIndices = []
countThresh = 0
indSort = np.argsort(counts)[::-1]
unique = unique[indSort]
counts = counts[indSort]
for i in range(len(unique)):
    if counts[i] >= countThresh:
        temp = np.where(idx==unique[i])[0]
        for j in range(len(temp)):
            newIndices.append(temp[j])
constraintsAllReducedUnique_organizedTemp = constraintsAllReducedUnique.iloc[newIndices,:]

if postSort:

    # within the clusters, sort by the number of mammals, birds, reptiles, and amphibians
    clusters = constraintsAllReducedUnique_organizedTemp['cluster'].unique()
    for i in range(len(clusters)):
        temp = constraintsAllReducedUnique_organizedTemp[constraintsAllReducedUnique_organizedTemp['cluster']==clusters[i]]
        # temp.sort_values(by=['numMammals','numBirds','numReptiles','numAmphibians'],inplace=True,ascending=False)
        temp.sort_values(by=['C','T','L','S','Ca'],inplace=True,ascending=False)
        if i == 0:
            constraintsAllReducedUnique_organized = temp
        else:
            constraintsAllReducedUnique_organized = pd.concat([constraintsAllReducedUnique_organized,temp],axis=0)

    constraintsAllReducedUnique_organized.reset_index(inplace=True,drop=True)
    
else:
    
    constraintsAllReducedUnique_organized = constraintsAllReducedUnique_organizedTemp

corrMatconstraintsAllReducedUnique_organized = corrMatrixFunction(constraintsAllReducedUnique_organized,1,6)
pdist_organized = spc.distance.pdist(corrMatconstraintsAllReducedUnique_organized)
linkage_organized = spc.linkage(pdist_organized, method=methodName)
idx_organized = spc.fcluster(linkage_organized, 0.5 * pdist_organized.max(), 'distance')


#%% some front matter before plots

vertebralData = pd.read_csv(outputPath+'vertebral/fullFormulaTree/vertebralData.csv')
# add the class
classList = pd.read_csv(outputPath+'vertebral/fullFormulaTree/class.csv',header=None)[0].to_list()
vertebralData['class'] = classList
mammals = vertebralData[vertebralData['class']=='Mammalia']
birds = vertebralData[vertebralData['class']=='Aves']
reptiles = vertebralData[vertebralData['class']=='Reptilia']
amphibians = vertebralData[vertebralData['class']=='Amphibia']
# add a tree species column (with the species name using an underscore)
speciesList = vertebralData['species'].to_list()
speciesList = [x.replace(' ','_') for x in speciesList]
vertebralData['treeSpecies'] = speciesList

indMammals = []
indBirds = []
indReptiles = []
indAmphibians = []
indMix = []
for i in range(len(PIC)):
    # mammals
    if ('Mammalia' in PIC['class'][i]) & ('Aves' not in PIC['class'][i]) & ('Reptilia' not in PIC['class'][i]) & ('Amphibia' not in PIC['class'][i]):
        indMammals.append(i)
    # birds
    elif ('Mammalia' not in PIC['class'][i]) & ('Aves' in PIC['class'][i]) & ('Reptilia' not in PIC['class'][i]) & ('Amphibia' not in PIC['class'][i]):
        indBirds.append(i)
    # reptiles
    elif ('Mammalia' not in PIC['class'][i]) & ('Aves' not in PIC['class'][i]) & ('Reptilia' in PIC['class'][i]) & ('Amphibia' not in PIC['class'][i]):
        indReptiles.append(i)
    # amphibians
    elif ('Mammalia' not in PIC['class'][i]) & ('Aves' not in PIC['class'][i]) & ('Reptilia' not in PIC['class'][i]) & ('Amphibia' in PIC['class'][i]):
        indAmphibians.append(i)
    else:
        indMix.append(i)
mammalsPIC = PIC.iloc[indMammals,:]
mammalsPIC = mammalsPIC.reset_index(drop=True)
birdsPIC = PIC.iloc[indBirds,:]
birdsPIC = birdsPIC.reset_index(drop=True)
reptilesPIC = PIC.iloc[indReptiles,:]
reptilesPIC = reptilesPIC.reset_index(drop=True)
amphibiansPIC = PIC.iloc[indAmphibians,:]
amphibiansPIC = amphibiansPIC.reset_index(drop=True)
mixPIC = PIC.iloc[indMix,:]
mixPIC = mixPIC.reset_index(drop=True)

#%% make another plot and just make tables of several choice constraints and plasticities

# reduce the font size
fontSize = 12
markerSize = 5

# Create a figure and a gridspec layout
fig = plt.figure(figsize=(18, 8))#,constrained_layout=False)
fig.subplots_adjust(hspace=2.5,wspace=2.0)
gs = gridspec.GridSpec(6,13)#,hspace=30000000)
ax0 = plt.subplot(gs[2:6,0:2])

ax4 = plt.subplot(gs[0:2,4:6]) # d
ax6 = plt.subplot(gs[0:2,6:8]) # e
ax8 = plt.subplot(gs[0:2,8:10]) # f
ax9 = plt.subplot(gs[2:4,4:6]) # g
ax7 = plt.subplot(gs[2:4,6:8]) # h
ax10 = plt.subplot(gs[2:4,8:10]) # i
ax11 = plt.subplot(gs[4:6,4:6]) # j
ax3 = plt.subplot(gs[4:6,6:8]) # k
ax5 = plt.subplot(gs[4:6,8:10]) # l

# constraints
# make a table of the constraints (hand-picked)
# constraintName = (['C','T','L','S','Ca','T+L','T+L+S','L+S','3S+Ca','C-S','C+T-S-Ca'])
constraintName = (['C','T','L','S','Ca','T+L','L+S','3S+Ca','C-S','C+T-S-Ca'])
cList = ([1,0,0,0,0,0,0,0,1,1])
tList = ([0,1,0,0,0,1,0,0,0,1])
lList = ([0,0,1,0,0,1,1,0,0,0])
sList = ([0,0,0,1,0,0,1,3,-1,-1])
caList = ([0,0,0,0,1,0,0,1,0,-1])
# classList = (['0','0','0','0','0','0','0','0','0','0'])
classList = ([r"$\mathcal{M}$,$\mathcal{A}$",
              r"$\mathcal{M}$,$\mathcal{B}$,$\mathcal{R}$,$\mathcal{A}$",
              r"$\mathcal{M}$,$\mathcal{B}$+$\mathcal{R}$,$\mathcal{A}$",
              r"$\mathcal{M}$,$\mathcal{B}$+$\mathcal{R}$,$\mathcal{A}$",
              r"$\mathcal{B}$,$\mathcal{R}$,$\mathcal{A}$",
              r"$\mathcal{M}$,$\mathcal{R}$",
            #   r"$\mathcal{R}$",
              r"$\mathcal{M}$",
              r"$\mathcal{M}$",
              r"$\mathcal{M}$+$\mathcal{B}$+$\mathcal{R}$+$\mathcal{A}$",
              r"$\mathcal{B}$"])
constraintTable = pd.DataFrame({'name':constraintName,'C':cList,'T':tList,'L':lList,'S':sList,'Ca':caList,'Classes':classList})
constraintTable = constraintTable[['name','C','T','L','S','Ca','Classes']]


# plasticities
# make a table of the plasticities (hand-picked)
plasticityName = (['T','Ca','C+S'])
cList = ([0,0,1])
tList = ([1,0,0])
lList = ([0,0,0])
sList = ([0,0,1])
caList = ([0,1,0])
classList = ([r"$\mathcal{M}$+$\mathcal{B}$+$\mathcal{R}$",
              r"$\mathcal{M}$,$\mathcal{R}$,$\mathcal{A}$",
              r"$\mathcal{B}$"])
# classList = ([r'$\mathcal{M}$,$\mathcal{A}','0','0','0','0','0','0','0','0','0'])
plasticityTable = pd.DataFrame({'name':plasticityName,'C':cList,'T':tList,'L':lList,'S':sList,'Ca':caList,'Classes':classList})
plasticityTable = plasticityTable[['name','C','T','L','S','Ca','Classes']]

# make a blank to insert between them
blank = pd.DataFrame({'name':[''],'C':[0],'T':[0],'L':[0],'S':[0],'Ca':[0],'Classes':['']})

constraintTable = pd.concat([constraintTable,blank,plasticityTable],axis=0)

# plot the table
# next plot the constraints themselves (their formula)
im0 = ax0.imshow(constraintTable.iloc[:,1:6],cmap=colorMap,interpolation='nearest',aspect=1,vmin=-3,vmax=3)
# put the text of the constraint in the middle of the cell if it is nonzero
for i in range(len(constraintTable)):
    for j in range(1,6):
        if constraintTable.iloc[i,j] != 0:
            ax0.text(j-1,i-0.0,constraintTable.iloc[i,j],fontsize=fontSize,color='k',horizontalalignment='center',verticalalignment='center')
# plot a "title" for the table
# put the constraintName as the label on the left
ax0.set_yticks([])
for i in range(len(constraintTable)):
    ax0.text(-0.9,i-0.0,constraintTable.iloc[i,0],fontsize=fontSize,color='k',horizontalalignment='right',verticalalignment='center')
# ax0.set_yticks(np.arange(len(constraintTable)))
# ax0.set_yticklabels(constraintName,fontsize=fontSize)
# put the Classes as the label on the right (y-label)
# so make another y-axis on the right without messing up the table
ax0.text(6,-1,'Class',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
# loop through the classList and plot M in 5, B in 6, R in 7, A in 8, if present
# and use the same color as the colorWheel (0,1,2,3,4) and do it as r"$\mathcal{M}$, etc.
for i in range(len(constraintTable)):
    if 'M' in constraintTable.iloc[i,6]:
        ax0.text(5,i-0.0,r"$\mathcal{M}$",fontsize=fontSize-1,color=colorWheel[0],horizontalalignment='center',verticalalignment='center')
    if 'B' in constraintTable.iloc[i,6]:
        ax0.text(5.75,i-0.0,r"$\mathcal{B}$",fontsize=fontSize-1,color=colorWheel[1],horizontalalignment='center',verticalalignment='center')
    if 'R' in constraintTable.iloc[i,6]:
        ax0.text(6.5,i-0.0,r"$\mathcal{R}$",fontsize=fontSize-1,color=colorWheel[2],horizontalalignment='center',verticalalignment='center')
    if 'A' in constraintTable.iloc[i,6]:
        ax0.text(7.25,i-0.0,r"$\mathcal{A}$",fontsize=fontSize-1,color=colorWheel[3],horizontalalignment='center',verticalalignment='center')

# remove the x-axis ticks
ax0.set_xticks([])

# draw lines to distinguish between the constraints and plasticities
# constraints
xPos0 = 8.5
xTextPos0 = 9.5
xTextPos0b = 9.5
ax0.annotate('', xy=(xPos0, -0.5), xycoords='data', xytext=(xPos0, 9.5), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax0.text(xTextPos0,5.2,'constraints',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# # plasticities
ax0.annotate('', xy=(xPos0, 10.5), xycoords='data', xytext=(xPos0, 13.5), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax0.text(xTextPos0b,11.9,'plasticities',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)

xPos = 11 # -0.6
xTestPos = 12 # -4
# # put the "types" on the left: constraints
ax0.annotate('', xy=(xPos, -0.5), xycoords='data', xytext=(xPos, 4.45), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax0.text(xTestPos,2,'$I$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# # type-II
ax0.annotate('', xy=(xPos, 4.5), xycoords='data', xytext=(xPos, 7.45), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax0.text(xTestPos,6.0,'$II$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
ax0.annotate('', xy=(xPos, 7.5), xycoords='data', xytext=(xPos, 9.5), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax0.text(xTestPos,8.5,'$III$',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)

# also put C,T,L,S,Ca on the top
ax0.text(0,-1,'C',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(1,-1,'T',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(2,-1,'L',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(3,-1,'S',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(4,-1,'Ca',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
# and on the bottom
ax0.text(0,14,'C',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(1,14,'T',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(2,14,'L',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(3,14,'S',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')
ax0.text(4,14,'Ca',fontsize=fontSize,horizontalalignment='center',verticalalignment='center')

# constraint: C-S = 0 (so plot S vs. C)
r,p = scipy.stats.pearsonr(vertebralData['Cervical'],vertebralData['Sacral'])
rPIC,pPIC = scipy.stats.pearsonr(PIC['C'],PIC['S'])
ax3.plot(mammals['Cervical'],mammals['Sacral'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax3.plot(birds['Cervical'],birds['Sacral'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax3.plot(reptiles['Cervical'],reptiles['Sacral'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label='Reptiles')
ax3.plot(amphibians['Cervical'],amphibians['Sacral'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label='Amphibians')
ax3.set_xlabel('Cervical',fontsize=fontSize)
ax3.set_ylabel('Sacral',fontsize=fontSize)
ax3.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# plot C=S line
ax3.plot([0,25],[0,25],'k--',linewidth=lineWidth)
# add text for C=S
ax3.text(0.55,0.83,'C=S',transform=ax3.transAxes,fontsize=fontSize)
# make tick labels smaller
ax3.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
saveForPIC(vertebralData['species'].to_list(),'Cervical','Sacral',vertebralData,tree,'_full')


# constraint and plasticity : C vs. Ca
r,p = scipy.stats.pearsonr(vertebralData['Cervical'],vertebralData['Caudal'])
ax4.plot(amphibians['Cervical'],amphibians['Caudal'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label=r'$\mathcal{A}$ (Amphibia)')
ax4.plot(mammals['Cervical'],mammals['Caudal'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label=r'$\mathcal{M}$ (Mammalia)')
ax4.plot(birds['Cervical'],birds['Caudal'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label=r'$\mathcal{B}$ (Aves)')
ax4.plot(reptiles['Cervical'],reptiles['Caudal'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label=r'$\mathcal{R}$ (Reptilia)')
ax4.set_xlabel('Cervical',fontsize=fontSize)
ax4.set_ylabel('Caudal',fontsize=fontSize)
ax4.tick_params(labelsize=fontSize-1)
ax4.set_xlim([0,30])
ax4.set_ylim([-5,190])
# flip the legend label and handles
ax4.annotate(r'$\mathcal{A}$ (Amphibia)',
    xy=(28,160), xycoords='data',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[3],ha='right')
ax4.annotate(r'$\mathcal{M}$ (Mammalia)',
    xy=(28,135), xycoords='data',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right')
ax4.annotate(r'$\mathcal{B}$ (Aves)',
    xy=(28,110), xycoords='data',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right')
ax4.annotate(r'$\mathcal{R}$ (Reptilia)',
    xy=(28,85), xycoords='data',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[2],ha='right')

# constraint S+Ca vs. C+T
# ax2 = plt.subplot(gs[0,2])
r,p = scipy.stats.pearsonr(vertebralData['Cervical']+vertebralData['Thoracic'],vertebralData['Sacral']+vertebralData['Caudal'])
rPIC,pPIC = scipy.stats.pearsonr(PIC['C']+PIC['T'],PIC['S']+PIC['Ca'])
rBirds,pBirds = scipy.stats.pearsonr(birds['Cervical']+birds['Thoracic'],birds['Sacral']+birds['Caudal'])
rPICbirds,pPICbirds = scipy.stats.pearsonr(birdsPIC['C']+birdsPIC['T'],birdsPIC['S']+birdsPIC['Ca'])
# ax2.plot(mammals['Cervical']+mammals['Thoracic'],mammals['Sacral']+mammals['Caudal'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax5.plot(birds['Cervical']+birds['Thoracic'],birds['Sacral']+birds['Caudal'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
# ax2.plot(reptiles['Cervical']+reptiles['Thoracic'],reptiles['Sacral']+reptiles['Caudal'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label='Reptiles')
# ax2.plot(amphibians['Cervical']+amphibians['Thoracic'],amphibians['Sacral']+amphibians['Caudal'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label='Amphibians')
ax5.set_xlabel('Cervical+Thoracic',fontsize=fontSize) 
ax5.set_ylabel('Sacral+Caudal',fontsize=fontSize)
ax5.plot([10,31],[10,31],'k--',linewidth=lineWidth)
ax5.text(0.05,0.6,'C+T=\nS+Ca',transform=ax5.transAxes,fontsize=fontSize)
# ax5.set_title('r='+str(round(rBirds,2))+', p='+"{:.2e}".format(pBirds)+"\nPIC: r="+str(round(rPICbirds,2))+', p='+"{:.2e}".format(pPICbirds),fontsize=fontSize)
# ax5.set_title('r='+str(round(rBirds,2))+'***, PIC: r='+str(round(rPICbirds,2))+'***',fontsize=fontSize)
ax5.set_title('r='+str(round(rBirds,2))+getAsterisks(pBirds)+', PIC: r='+str(round(rPICbirds,2))+getAsterisks(pPICbirds),fontsize=fontSize)
ax5.tick_params(labelsize=fontSize-1)
print(rBirds,pBirds,rPICbirds,pPICbirds)
birds['Cervical+Thoracic'] = birds['Cervical']+birds['Thoracic']
birds['Sacral+Caudal'] = birds['Sacral']+birds['Caudal']
saveForPIC(birds['species'].to_list(),'Cervical+Thoracic','Sacral+Caudal',birds,tree,'_birds')

# constraint and plasticity : C+S vs. T

ax6.plot(amphibians['Cervical']+amphibians['Sacral'],amphibians['Thoracic'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label=r'$\mathcal{A}$ (Amphibia)')
ax6.plot(mammals['Cervical']+mammals['Sacral'],mammals['Thoracic'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label=r'$\mathcal{M}$ (Mammalia)')
ax6.plot(birds['Cervical']+birds['Sacral'],birds['Thoracic'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label=r'$\mathcal{B}$ (Aves)')
ax6.plot(reptiles['Cervical']+reptiles['Sacral'],reptiles['Thoracic'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label=r'$\mathcal{R}$ (Reptilia)')

ax6.set_xlabel('Cervical+Sacral',fontsize=fontSize)
ax6.set_ylabel('Thoracic',fontsize=fontSize)
ax6.tick_params(labelsize=fontSize-1)
# ax6.legend(frameon=False,fontsize=fontSize-1,loc=([0.12,0.42]),labelspacing=0.05,handletextpad=-0.25,markerfirst=False)
ax6.set_xlim([-1,60])
# make the significance block
# add inset on top right that extends outside the figure axis on the top and the right
axins = ax6.inset_axes([0.4,0.5,0.65,0.55])
# remove all ticks
axins.set_xticks([])
axins.set_yticks([])
# annotate:
# "p-value
# < 0.05 *
# < 0.01 **
# < 0.001 ***"
axins.annotate('p-value\n< 0.05 *\n< 0.01 **\n< 0.001 ***',
    xy=(0.08,0.88), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',ha='left',va='top',fontsize=fontSize)


# constraint: S vs. Ca in mammals
# ax4 = plt.subplot(gs[1,1])
r,p = scipy.stats.pearsonr(mammals['Caudal'],mammals['Sacral'])
rPIC,pPIC = scipy.stats.pearsonr(mammalsPIC['Ca'],mammalsPIC['S'])
ax7.plot(mammals['Sacral'],mammals['Caudal'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax7.set_xlabel('Sacral',fontsize=fontSize)
ax7.set_ylabel('Caudal',fontsize=fontSize)
ax7.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax7.set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+"\nPIC: r="+str(round(rPIC,2))+', p='+"{:.2e}".format(pPIC),fontsize=fontSize)
# plot a fit line
m = np.polyfit(mammals['Sacral'],mammals['Caudal'],1)
xi = np.linspace(np.nanmin(mammals['Sacral']),np.nanmax(mammals['Sacral']),100)
# ax7.plot(xi,m[0]*xi+m[1],'k--',linewidth=lineWidth)
# ax7.text(0.3,0.15,'S='+str(round(m[0],2))+'Ca+'+str(round(m[1],2)),transform=ax7.transAxes,fontsize=fontSize)
ax7.tick_params(labelsize=fontSize-1)
ax7.set_ylim([-5,43])
ax7.set_xlim([-1,10])
print(r,p,rPIC,pPIC)
saveForPIC(mammals['species'].to_list(),'Sacral','Caudal',mammals,tree,'_mammals')


# plot L vs. T in mammals
# ax6 = plt.subplot(gs[2,0])
r,p = scipy.stats.pearsonr(mammals['Thoracic'],mammals['Lumbar'])
rPIC,pPIC = scipy.stats.pearsonr(mammalsPIC['T'],mammalsPIC['L'])
ax8.plot(mammals['Thoracic'],mammals['Lumbar'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax8.set_xlabel('Thoracic',fontsize=fontSize)
ax8.set_ylabel('Lumbar',fontsize=fontSize)
ax8.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax8.set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+"\nPIC: r="+str(round(rPIC,2))+', p='+"{:.2e}".format(pPIC),fontsize=fontSize)
# plot a fit line
m = np.polyfit(mammals['Thoracic'],mammals['Lumbar'],1)
xi = np.linspace(np.nanmin(mammals['Thoracic']),np.nanmax(mammals['Thoracic']),100)
# ax8.plot(xi,m[0]*xi+m[1],'k--',linewidth=lineWidth)
# ax8.text(0.3,0.525,'T='+str(round(m[0],2))+'L+'+str(round(m[1],2)),transform=ax8.transAxes,fontsize=fontSize)
ax8.tick_params(labelsize=fontSize-1)
ax8.set_ylim([0,13])
ax8.set_xlim([8,25])
ax8.set_xticks([10,15,20,25])
print(r,p,rPIC,pPIC)
saveForPIC(mammals['species'].to_list(),'Thoracic','Lumbar',mammals,tree,'_mammals')

# plot S vs. L in mammals
# ax6 = plt.subplot(gs[2,0])
r,p = scipy.stats.pearsonr(mammals['Lumbar'],mammals['Sacral'])
rPIC,pPIC = scipy.stats.pearsonr(mammalsPIC['L'],mammalsPIC['S'])
ax9.plot(mammals['Lumbar'],mammals['Sacral'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax9.set_xlabel('Lumbar',fontsize=fontSize)
ax9.set_ylabel('Sacral',fontsize=fontSize)
ax9.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax8.set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+"\nPIC: r="+str(round(rPIC,2))+', p='+"{:.2e}".format(pPIC),fontsize=fontSize)
# plot a fit line
m = np.polyfit(mammals['Lumbar'],mammals['Sacral'],1)
xi = np.linspace(np.nanmin(mammals['Lumbar']),np.nanmax(mammals['Lumbar']),100)
# ax9.plot(xi,m[0]*xi+m[1],'k--',linewidth=lineWidth)
# ax8.text(0.3,0.525,'T='+str(round(m[0],2))+'L+'+str(round(m[1],2)),transform=ax8.transAxes,fontsize=fontSize)
ax9.tick_params(labelsize=fontSize-1)
ax9.set_ylim([-2,9])
ax9.set_xlim([1,15])
ax9.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
saveForPIC(mammals['species'].to_list(),'Lumbar','Sacral',mammals,tree,'_mammals')


# load williams data
filePath = inputPath+'additionalData/williamsPresacralIncreasedVariation_supplementaryTable1.csv'
williams = pd.read_csv(filePath)
# remove the last few rows
williams = williams.iloc[:-18,:]
# replace all the '-' with np.nan
williams = williams.replace('-',np.nan)
# replace 'nan' with np.nan
williams = williams.replace('nan',np.nan)
# make the 'S mode' column numeric
williams['S Mode'] = pd.to_numeric(williams['S Mode'])
# rename 'S Mode' to 'Sacral' and 'L Mode' to 'Lumbar'
williams = williams.rename(columns={'S Mode':'Sacral','L Mode':'Lumbar'})
# plot the williams data
# average the numerical columns by the genus
williamsGenus = williams['Genus'].unique()
williamsLumbar = []
williamsSacral = []
for genus in williamsGenus:
    williamsLumbar.append(williams[williams['Genus'] == genus]['Lumbar'].mean())
    williamsSacral.append(williams[williams['Genus'] == genus]['Sacral'].mean())
ax9.plot(williamsLumbar,williamsSacral,'o',markersize=markerSize,color=colorWheel[0],mfc='None',alpha=0.1,label='Williams et al.')

# williams without nans in Sacral
williamsNoNan = williams[~np.isnan(williams['Sacral'])]

# load cetacean data
filePathCetaceans = inputPath+'additionalData/buchholtzCetaceans.csv'
buchholtz = pd.read_csv(filePathCetaceans)
buchholtzGenus = buchholtz['genus'].unique()
buchholtzLumbar = []
buchholtzSacral = []
for genus in buchholtzGenus:
    buchholtzLumbar.append(buchholtz[buchholtz['genus'] == genus]['L'].mean())
    buchholtzSacral.append(buchholtz[buchholtz['genus'] == genus]['S'].mean())
ax9.plot(buchholtzLumbar,buchholtzSacral,'o',markersize=markerSize,color=colorWheel[0],mfc='None',alpha=0.1,label='Buchholtz et al.')

# combine williams and our mammals and get the correlation
xLumbar = pd.concat([mammals['Lumbar'],williams['Lumbar'],buchholtz['L']])
xSacral = pd.concat([mammals['Sacral'],williams['Sacral'],buchholtz['S']])
# add a genus column to mammals if not present already
if 'genus' not in mammals.columns:
    mammals['genus'] = mammals['species'].apply(lambda x: x.split(' ')[0])
genus = pd.concat([mammals['genus'],williams['Genus'],buchholtz['genus']])
# make a new dataframe and remove any rows with nans
combined = pd.DataFrame({'Lumbar':xLumbar,'Sacral':xSacral,'Genus':genus})
combined = combined.dropna()
# average over the genera
combined = combined.groupby('Genus').mean().reset_index()
r,p = scipy.stats.pearsonr(combined['Lumbar'],combined['Sacral'])
print('r,p for combined williams, buchholtz, and our mammals:',r,p)

# note the Orca
ax9.text(mammals['Lumbar'][mammals['species']=='Orcinus orca'].iloc[0]-3.0,mammals['Sacral'][mammals['species']=='Orcinus orca'].iloc[0]-1.5,'Cetaceans',fontsize=fontSize-1)
# ax9.text(5.4,7.0,'non-Cetaceans',fontsize=fontSize-1)
# draw an oval around the cetaceans
patches = mpl.patches
oval = patches.Ellipse((mammals['Lumbar'][mammals['species']=='Orcinus orca'].iloc[0],mammals['Sacral'][mammals['species']=='Orcinus orca'].iloc[0]-0.0),width=22,height=1.1,edgecolor='black',facecolor='none',linewidth=0.5,alpha=0.5)
ax9.add_patch(oval)

# plot mammals everything (C+T+2L+2S) vs. Ca/3
# or C+T+L+S+Ca vs. Ca
r,p = scipy.stats.pearsonr(mammals['Cervical']+1*mammals['Thoracic']+1*mammals['Lumbar']+1*mammals['Sacral'],mammals['Caudal'])
ax10.plot(mammals['Caudal'],mammals['Cervical']+1*mammals['Thoracic']+1*mammals['Lumbar']+1*mammals['Sacral'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')

ax10.set_ylabel('pre-Caudal',fontsize=fontSize)
ax10.set_xlabel('Caudal',fontsize=fontSize)
# ax10.set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p),fontsize=fontSize)
rPIC,pPIC = scipy.stats.pearsonr(mammalsPIC['C']+mammalsPIC['T']+1*mammalsPIC['L']+1*mammalsPIC['S'],mammalsPIC['Ca']/1)
ax10.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# plot a line S = L
ax10.set_ylim([0,45])
ax10.set_xlim([0,45])
# ax10.set_xlim([7.5,11])
ax10.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
mammals['preCaudal'] = mammals['Cervical']+mammals['Thoracic']+mammals['Lumbar']+mammals['Sacral']
saveForPIC(mammals['species'].to_list(),'Caudal','preCaudal',mammals,tree,'_mammals')

# load the species from node 1351 and plot S vs. L
species1351 = pd.read_csv(outputPath+'vertebral/subTree_1351_1351_speciesNum146/species.csv',header=None)[0].to_list()
# only plot these species (reptiles)
# ax5 = plt.subplot(gs[1,2])
r,p = scipy.stats.pearsonr(vertebralData['Cervical'][vertebralData['species'].isin(species1351)],vertebralData['Thoracic'][vertebralData['species'].isin(species1351)])
ax11.plot(birds['Cervical'][birds['species'].isin(species1351)],birds['Thoracic'][birds['species'].isin(species1351)],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax11.plot(reptiles['Cervical'][reptiles['species'].isin(species1351)],reptiles['Thoracic'][reptiles['species'].isin(species1351)],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label='Reptiles')
ax11.set_xlabel('Cervical',fontsize=fontSize)
ax11.set_ylabel('Thoracic',fontsize=fontSize)
# ax11.set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p),fontsize=fontSize)
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0]
# ax11.set_title('r='+str(round(r,2))+'***, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0],2))+'***',fontsize=fontSize)
ax11.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)

ax11.set_ylim([2,26])
ax11.set_xlim([-1,25])
ax11.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
# fit to all in species1351
m = np.polyfit(vertebralData['Cervical'][vertebralData['species'].isin(species1351)],vertebralData['Thoracic'][vertebralData['species'].isin(species1351)],1)
xi = np.linspace(np.nanmin(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),np.nanmax(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),100)
# ax11.plot(xi,m[0]*xi+m[1],'k--',linewidth=lineWidth)
ax11.text(0.95,0.67,'Testudinata,\nCrocodilia,\nAves',transform=ax11.transAxes,fontsize=fontSize-1,ha='right')
# plot all
# mammals
ax11.plot(mammals['Cervical'],mammals['Thoracic'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.05,label='Mammals')
# birds
ax11.plot(birds['Cervical'],birds['Thoracic'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.05,label='Birds')
# reptiles
ax11.plot(reptiles['Cervical'],reptiles['Thoracic'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.05,label='Reptiles')
# amphibians
ax11.plot(amphibians['Cervical'],amphibians['Thoracic'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.05,label='Amphibians')
# get the correlation for everything excluding amphibians and snakes
r,p = scipy.stats.pearsonr(vertebralData['Cervical'][(~vertebralData['species'].isin(amphibians['species']))&(vertebralData['Thoracic']<50)],vertebralData['Thoracic'][(~vertebralData['species'].isin(amphibians['species']))&(vertebralData['Thoracic']<50)])
print(r,p)
saveForPIC(species1351,'Cervical','Thoracic',vertebralData,tree,'_node1351')


# plot the A, B, etc. labels
ax0.text(-.5,1.615,'A',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)
ax0.text(-.5,1.03,'B',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)
ax3.text(-0.3,1.03,'J',fontsize=fontSize+4,fontweight='normal',transform=ax3.transAxes)
ax4.text(-0.3,1.03,'C',fontsize=fontSize+4,fontweight='normal',transform=ax4.transAxes)
ax5.text(-0.3,1.03,'K',fontsize=fontSize+4,fontweight='normal',transform=ax5.transAxes)
ax6.text(-0.3,1.03,'D',fontsize=fontSize+4,fontweight='normal',transform=ax6.transAxes)
ax7.text(-0.3,1.03,'G',fontsize=fontSize+4,fontweight='normal',transform=ax7.transAxes)
ax8.text(-0.3,1.03,'E',fontsize=fontSize+4,fontweight='normal',transform=ax8.transAxes)
ax9.text(-0.3,1.03,'F',fontsize=fontSize+4,fontweight='normal',transform=ax9.transAxes)
ax10.text(-0.3,1.03,'H',fontsize=fontSize+4,fontweight='normal',transform=ax10.transAxes)
ax11.text(-0.3,1.03,'I',fontsize=fontSize+4,fontweight='normal',transform=ax11.transAxes)

plt.savefig(outputPath+'plots/constraintsPlasticitiesExamples_Fig2_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/constraintsPlasticitiesExamples_Fig2_v2.pdf',dpi=300,bbox_inches='tight')

#%% determine which constraints are both "local" and "global"

constraintThreshold = 0.05 # inside and out

print('constraints:')

indSuper = []
for i in range(len(constraintsAllReducedUnique_organized)):
    if (constraintsAllReducedUnique_organized['insideVariance'].iloc[i] < constraintThreshold*constraintsAllReducedUnique_organized['totalVariance'].iloc[i]) & (constraintsAllReducedUnique_organized['insideVariance'].iloc[i] < constraintThreshold*constraintsAllReducedUnique_organized['outsideVariance'].iloc[i]):
        indSuper.append(i)
        print(constraintsAllReducedUnique_organized.iloc[i].to_list())
indSuper = np.array(indSuper)

print('plasticities:')

# and now with the plasticities

plasticityThresholdLocal = 0.8 # inside and out"
plasticityThresholdGlobal = 0.9 # inside and out
for i in range(len(plasticities)):
    if (plasticities['insideVariance'].iloc[i] > plasticityThresholdLocal*plasticities['totalVariance'].iloc[i]) & (plasticities['insideVariance'].iloc[i] > plasticityThresholdGlobal*plasticities['outsideVariance'].iloc[i]):
        print(plasticities.iloc[i].to_list())
        
        
#%% make a table of all the constraints and plasticities
# sort by those with a single non-zero coefficient first
# then the adjacent
# then the non-adjacent
# then those that don't fit into any of these categories: "outsiders"
# and within each of these three groups sort by C, then T, then L, then S, then Ca

constraintsAllOrganized = constraintsAllReducedUnique_organized.copy()
# get the number of non-zero coefficients
numNonZero = []
for i in range(len(constraintsAllOrganized)):
    numNonZero.append(np.count_nonzero(constraintsAllOrganized.iloc[i,1:6]))
constraintsAllOrganized['numNonZero'] = numNonZero
# sort by numNonZero
# constraintsAllOrganized = constraintsAllOrganized.sort_values(by=['numNonZero','Ca','S','L','T','C'],ascending=True)
constraintsSingle = constraintsAllOrganized[constraintsAllOrganized['numNonZero']==1]
constraintsSingle = constraintsSingle.sort_values(by=['Ca','S','L','T','C'],ascending=True)

constraintsPlural = constraintsAllOrganized[constraintsAllOrganized['numNonZero']>1]
# find the adjacent and non-adjacent ones
# this is a function suggested by chatgpt which I modified

def categorize_rows(array):
    adjacent = []

    for row in array:
        nonzeros = np.nonzero(row)[0]

        if all(np.diff(nonzeros) == 1):
            adjacent.append(1)
        else:
            adjacent.append(0)

    return adjacent

adjacent = categorize_rows(constraintsPlural.iloc[:,1:6].to_numpy())
constraintsPlural['adjacent'] = adjacent
constraintsPluralAdjacent = constraintsPlural[constraintsPlural['adjacent']==1]
constraintsPluralNonAdjacent = constraintsPlural[constraintsPlural['adjacent']==0]
# sort the adjacent ones by the vertebrae
constraintsPluralAdjacent = constraintsPluralAdjacent.sort_values(by=['Ca','S','L','T','C'],ascending=True)
# if there are any adjacent with any negative coefficients, take them out and put them in a separate category of "outsiders"
constraintsPluralAdjacentOutsiders = constraintsPluralAdjacent[(constraintsPluralAdjacent['C']<0)|(constraintsPluralAdjacent['T']<0)|(constraintsPluralAdjacent['L']<0)|(constraintsPluralAdjacent['S']<0)|(constraintsPluralAdjacent['Ca']<0)]
# remove those from the adjacent
constraintsPluralAdjacent = constraintsPluralAdjacent[~constraintsPluralAdjacent.index.isin(constraintsPluralAdjacentOutsiders.index)]
# sort the non-adjacent ones by the vertebrae
constraintsPluralNonAdjacent = constraintsPluralNonAdjacent.sort_values(by=['Ca','S','L','T','C'],ascending=True)
# if there are any non-adjacent with only positive coefficients, take them out and put them in a separate category of "outsiders"
constraintsPluralNonAdjacentOutsiders = constraintsPluralNonAdjacent[(constraintsPluralNonAdjacent['C']>=0)&(constraintsPluralNonAdjacent['T']>=0)&(constraintsPluralNonAdjacent['L']>=0)&(constraintsPluralNonAdjacent['S']>=0)&(constraintsPluralNonAdjacent['Ca']>=0)]
# remove these from the non-adjacent
constraintsPluralNonAdjacent = constraintsPluralNonAdjacent[~constraintsPluralNonAdjacent.index.isin(constraintsPluralNonAdjacentOutsiders.index)]
# recombine all the constraints
constraintsAllOrganizedRecombined = pd.concat([constraintsSingle,constraintsPluralAdjacent,constraintsPluralNonAdjacent,constraintsPluralAdjacentOutsiders,constraintsPluralNonAdjacentOutsiders])
# reset the index
constraintsAllOrganizedRecombined = constraintsAllOrganizedRecombined.reset_index(drop=True)
# get the lengths of each for future reference
numSingle = len(constraintsSingle)
numPluralAdjacent = len(constraintsPluralAdjacent)
numPluralNonAdjacent = len(constraintsPluralNonAdjacent)
numPluralAdjacentOutsiders = len(constraintsPluralAdjacentOutsiders)
numPluralNonAdjacentOutsiders = len(constraintsPluralNonAdjacentOutsiders)
numConstraints = len(constraintsAllOrganizedRecombined)

# make names for the constraints and also put the constraint name as the first column
constraintName = []
for i in range(len(constraintsAllOrganizedRecombined)):
    temp = ''
    if constraintsAllOrganizedRecombined['C'].iloc[i] != 0:
        if constraintsAllOrganizedRecombined['C'].iloc[i] > 1:
            temp = temp + '+' + str(int(constraintsAllOrganizedRecombined['C'].iloc[i])) + 'C'
        else:
            temp = temp + '+C'
    if constraintsAllOrganizedRecombined['T'].iloc[i] != 0:
        if constraintsAllOrganizedRecombined['T'].iloc[i] > 1:
            temp = temp + '+' + str(int(constraintsAllOrganizedRecombined['T'].iloc[i])) + 'T'
        elif constraintsAllOrganizedRecombined['T'].iloc[i] < -1:
            temp = temp + str(int(constraintsAllOrganizedRecombined['T'].iloc[i])) + 'T'
        elif constraintsAllOrganizedRecombined['T'].iloc[i] == 1:
            temp = temp + '+T'
        else:
            temp = temp + '-T'
    if constraintsAllOrganizedRecombined['L'].iloc[i] != 0:
        if constraintsAllOrganizedRecombined['L'].iloc[i] > 1:
            temp = temp + '+' + str(int(constraintsAllOrganizedRecombined['L'].iloc[i])) + 'L'
        elif constraintsAllOrganizedRecombined['L'].iloc[i] < -1:
            temp = temp + str(int(constraintsAllOrganizedRecombined['L'].iloc[i])) + 'L'
        elif constraintsAllOrganizedRecombined['L'].iloc[i] == 1:
            temp = temp + '+L'
        else:
            temp = temp + '-L'
    if constraintsAllOrganizedRecombined['S'].iloc[i] != 0:
        if constraintsAllOrganizedRecombined['S'].iloc[i] > 1:
            temp = temp + '+' + str(int(constraintsAllOrganizedRecombined['S'].iloc[i])) + 'S'
        elif constraintsAllOrganizedRecombined['S'].iloc[i] < -1:
            temp = temp + str(int(constraintsAllOrganizedRecombined['S'].iloc[i])) + 'S'
        elif constraintsAllOrganizedRecombined['S'].iloc[i] == 1:
            temp = temp + '+S'
        else:
            temp = temp + '-S'
    if constraintsAllOrganizedRecombined['Ca'].iloc[i] != 0:
        if constraintsAllOrganizedRecombined['Ca'].iloc[i] > 1:
            temp = temp + '+' + str(int(constraintsAllOrganizedRecombined['Ca'].iloc[i])) + 'Ca'
        elif constraintsAllOrganizedRecombined['Ca'].iloc[i] < -1:
            temp = temp + str(int(constraintsAllOrganizedRecombined['Ca'].iloc[i])) + 'Ca'
        elif constraintsAllOrganizedRecombined['Ca'].iloc[i] == 1:
            temp = temp + '+Ca'
        else:
            temp = temp + '-Ca'
    # remove the first plus sign
    temp = temp[1:]
    constraintName.append(temp)
constraintsAllOrganizedRecombined['name'] = constraintName

# now do this with the plasticities

plasticitiesAllOrganized = plasticities.copy()
numNonZero = []
for i in range(len(plasticitiesAllOrganized)):
    numNonZero.append(np.count_nonzero(plasticitiesAllOrganized.iloc[i,1:6]))
plasticitiesAllOrganized['numNonZero'] = numNonZero
plasticitiesSingle = plasticitiesAllOrganized[plasticitiesAllOrganized['numNonZero']==1]
# sort
plasticitiesSingle = plasticitiesSingle.sort_values(by=['Ca','S','L','T','C'],ascending=True)

plasticitiesPlural = plasticitiesAllOrganized[plasticitiesAllOrganized['numNonZero']>1]
# find the adjacent and non-adjacent ones
adjacent = categorize_rows(plasticitiesPlural.iloc[:,1:6].to_numpy())
plasticitiesPlural['adjacent'] = adjacent
plasticitiesPluralAdjacent = plasticitiesPlural[plasticitiesPlural['adjacent']==1]
plasticitiesPluralNonAdjacent = plasticitiesPlural[plasticitiesPlural['adjacent']==0]
# sort the adjacent ones by the vertebrae
plasticitiesPluralAdjacent = plasticitiesPluralAdjacent.sort_values(by=['Ca','S','L','T','C'],ascending=True)
# sort the non-adjacent ones by the vertebrae
plasticitiesPluralNonAdjacent = plasticitiesPluralNonAdjacent.sort_values(by=['Ca','S','L','T','C'],ascending=True)
# recombine
plasticitiesAllOrganizedRecombined = pd.concat([plasticitiesSingle,plasticitiesPluralAdjacent,plasticitiesPluralNonAdjacent])
# reset the index
plasticitiesAllOrganizedRecombined = plasticitiesAllOrganizedRecombined.reset_index(drop=True)


# make names for the plasticities and also put the plasticity name as the first column
plasticityName = []
for i in range(len(plasticitiesAllOrganizedRecombined)):
    temp = ''
    if plasticitiesAllOrganizedRecombined['C'].iloc[i] != 0:
        if plasticitiesAllOrganizedRecombined['C'].iloc[i] > 1:
            temp = temp + '+' + str(int(plasticitiesAllOrganizedRecombined['C'].iloc[i])) + 'C'
        else:
            temp = temp + '+C'
    if plasticitiesAllOrganizedRecombined['T'].iloc[i] != 0:
        if plasticitiesAllOrganizedRecombined['T'].iloc[i] > 1:
            temp = temp + '+' + str(int(plasticitiesAllOrganizedRecombined['T'].iloc[i])) + 'T'
        elif plasticitiesAllOrganizedRecombined['T'].iloc[i] < -1:
            temp = temp + str(int(plasticitiesAllOrganizedRecombined['T'].iloc[i])) + 'T'
        elif plasticitiesAllOrganizedRecombined['T'].iloc[i] == 1:
            temp = temp + '+T'
        else:
            temp = temp + '-T'
    if plasticitiesAllOrganizedRecombined['L'].iloc[i] != 0:
        if plasticitiesAllOrganizedRecombined['L'].iloc[i] > 1:
            temp = temp + '+' + str(int(plasticitiesAllOrganizedRecombined['L'].iloc[i])) + 'L'
        elif plasticitiesAllOrganizedRecombined['L'].iloc[i] < -1:
            temp = temp + str(int(plasticitiesAllOrganizedRecombined['L'].iloc[i])) + 'L'
        elif plasticitiesAllOrganizedRecombined['L'].iloc[i] == 1:
            temp = temp + '+L'
        else:
            temp = temp + '-L'
    if plasticitiesAllOrganizedRecombined['S'].iloc[i] != 0:
        if plasticitiesAllOrganizedRecombined['S'].iloc[i] > 1:
            temp = temp + '+' + str(int(plasticitiesAllOrganizedRecombined['S'].iloc[i])) + 'S'
        elif plasticitiesAllOrganizedRecombined['S'].iloc[i] < -1:
            temp = temp + str(int(plasticitiesAllOrganizedRecombined['S'].iloc[i])) + 'S'
        elif plasticitiesAllOrganizedRecombined['S'].iloc[i] == 1:
            temp = temp + '+S'
        else:
            temp = temp + '-S'
    if plasticitiesAllOrganizedRecombined['Ca'].iloc[i] != 0:
        if plasticitiesAllOrganizedRecombined['Ca'].iloc[i] > 1:
            temp = temp + '+' + str(int(plasticitiesAllOrganizedRecombined['Ca'].iloc[i])) + 'Ca'
        elif plasticitiesAllOrganizedRecombined['Ca'].iloc[i] < -1:
            temp = temp + str(int(plasticitiesAllOrganizedRecombined['Ca'].iloc[i])) + 'Ca'
        elif plasticitiesAllOrganizedRecombined['Ca'].iloc[i] == 1:
            temp = temp + '+Ca'
        else:
            temp = temp + '-Ca'
    # remove the first plus sign
    temp = temp[1:]
    plasticityName.append(temp)
plasticitiesAllOrganizedRecombined['name'] = plasticityName

# make a blank pandas dataframe to put in between the constraints and plasticities
blank = pd.DataFrame(np.zeros((1,17)),columns=['name','C','T','L','S','Ca','numMammals','numBirds','numReptiles','numAmphibians','insideVariance','outsideVariance','totalVariance','r','p','rPIC','pPIC'])
indBlank = len(constraintsAllOrganizedRecombined)

constraintReduced = constraintsAllOrganizedRecombined.copy()
# keep only the same columns as in "blank" above
constraintReduced = constraintReduced[['name','C','T','L','S','Ca','numMammals','numBirds','numReptiles','numAmphibians','insideVariance','outsideVariance','totalVariance','r','p','rPIC','pPIC']]

plasticityReduced = plasticitiesAllOrganizedRecombined.copy()
# add 'r', 'p', 'rPIC', and 'pPIC' columns
plasticityReduced['r'] = np.nan
plasticityReduced['p'] = np.nan
plasticityReduced['rPIC'] = np.nan
plasticityReduced['pPIC'] = np.nan
# keep only the same columns as in "blank" above
plasticityReduced = plasticityReduced[['name','C','T','L','S','Ca','numMammals','numBirds','numReptiles','numAmphibians','insideVariance','outsideVariance','totalVariance','r','p','rPIC','pPIC']]


# combine all

constraintsAndPlasticities = pd.concat([constraintReduced,blank,plasticityReduced])
# reset the index
constraintsAndPlasticities = constraintsAndPlasticities.reset_index(drop=True)

# double check that all single non-zero constraints have positive coefficients
for i in range(len(constraintsAndPlasticities)):
    numNonZeros = np.count_nonzero(constraintsAndPlasticities.iloc[i,1:6])
    if numNonZeros == 1:
        # get ind of that one nonzero (which may be less than zero)
        ind = np.where(constraintsAndPlasticities.iloc[i,1:6] != 0)[0][0]
        if constraintsAndPlasticities.iloc[i,ind+1] < 0:
            constraintsAndPlasticities.iloc[i,ind+1] = -constraintsAndPlasticities.iloc[i,ind+1]

# and finally double check that there are no duplicates in the node and the coefficients
# (so check if columns 0 to 10 are identical) then only keep the first one
constraintsAndPlasticities = constraintsAndPlasticities.drop_duplicates(subset=constraintsAndPlasticities.columns[0:10])
# reorganize the plasticities after the row with all zeros in the coefficients (the blank row)
# so that they start with the non-zero coefficients
# first find the index of the blank row
indBlank = constraintsAndPlasticities[constraintsAndPlasticities['name'] == 0].index[0]
# then reorganize the plasticities
plasticityOnly = constraintsAndPlasticities.iloc[indBlank+1:,:]
# then reorganize the plasticities
plasticityOnly = plasticityOnly.sort_values(by=['C','T','L','S','Ca'],ascending=False)
# put any with more than one non-zero coefficient at the end
numNonZero = []
for i in range(len(plasticityOnly)):
    numNonZero.append(np.count_nonzero(plasticityOnly.iloc[i,1:6]))
indMultipleNonZero = np.where(np.array(numNonZero) > 1)[0]
indOneNonZero = np.where(np.array(numNonZero) == 1)[0]
plasticityOneNonZero = plasticityOnly.iloc[indOneNonZero,:]
plasticityMultipleNonZero = plasticityOnly.iloc[indMultipleNonZero,:]
plasticityOnly = pd.concat([plasticityOneNonZero,plasticityMultipleNonZero])

# get the number of all plasticities for future reference
numPlasticities = len(plasticityOnly)

# recombine
constraintsAndPlasticities = pd.concat([constraintsAndPlasticities.iloc[:indBlank+1,:],plasticityOnly])
# reset index
constraintsAndPlasticities = constraintsAndPlasticities.reset_index(drop=True)

#%% plot the additional table

fig,ax = plt.subplots(figsize=(2,16))

# next plot the constraints themselves (their formula)
im0 = ax.imshow(constraintsAndPlasticities.iloc[:,1:6],cmap=colorMap,interpolation='nearest',aspect='auto',vmin=-3,vmax=3)

for i in range(len(constraintsAndPlasticities)):
    for j in range(1,6):
        if constraintsAndPlasticities.iloc[i,j] != 0:
            ax.text(j-1,i-0.0,int(constraintsAndPlasticities.iloc[i,j]),fontsize=fontSize,color='k',horizontalalignment='center',verticalalignment='center')
# plot a "title" for the table
# put the constraintName as the label on the left
ax.set_yticks([])

for i in range(indBlank):
    ax.text(-0.9,i-0.0,constraintsAndPlasticities.iloc[i,0],fontsize=fontSize,color='k',horizontalalignment='right',verticalalignment='center')
for i in range(indBlank+1,len(constraintsAndPlasticities)):
    ax.text(-0.9,i-0.0,constraintsAndPlasticities.iloc[i,0],fontsize=fontSize,color='k',horizontalalignment='right',verticalalignment='center')

# spacing
initialSpacing = 5.5
spacing = 3

# plot the number of species in each class as text using ax.text to the right
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+0*spacing,i-0.0,int(constraintsAndPlasticities.iloc[i,6]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+1*spacing,i-0.0,int(constraintsAndPlasticities.iloc[i,7]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+2*spacing,i-0.0,int(constraintsAndPlasticities.iloc[i,8]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+3*spacing,i-0.0,int(constraintsAndPlasticities.iloc[i,9]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
# next the variance, total variance, outside variance
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+4*spacing,i-0.0,str(round(constraintsAndPlasticities.iloc[i,10],2)),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
for i in range(len(constraintsAndPlasticities)):
    if (i == indBlank) | (np.isnan(constraintsAndPlasticities.iloc[i,11])):
        continue
    ax.text(initialSpacing+5*spacing,i-0.0,str(round(constraintsAndPlasticities.iloc[i,11],2)),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
for i in range(len(constraintsAndPlasticities)):
    if i == indBlank:
        continue
    ax.text(initialSpacing+6*spacing,i-0.0,str(round(constraintsAndPlasticities.iloc[i,12],2)),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')

# r, p, rPIC, pPIC
# for the p and pPIC this should be in scientific notation
# r
for i in range(len(constraintsAndPlasticities)):
    if (i == indBlank) | (np.isnan(constraintsAndPlasticities.iloc[i,13])):
        continue
    ax.text(initialSpacing+7*spacing,i-0.0,str(round(constraintsAndPlasticities.iloc[i,13],2)),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
# p (scientific notation)
for i in range(len(constraintsAndPlasticities)):
    if (i == indBlank) | (np.isnan(constraintsAndPlasticities.iloc[i,13])):
        continue
    ax.text(initialSpacing+8*spacing,i-0.0,"{:.2e}".format(constraintsAndPlasticities.iloc[i,14]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
# rPIC
for i in range(len(constraintsAndPlasticities)):
    if (i == indBlank) | (np.isnan(constraintsAndPlasticities.iloc[i,13])):
        continue
    ax.text(initialSpacing+9*spacing,i-0.0,str(round(constraintsAndPlasticities.iloc[i,15],2)),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')
# pPIC (scientific notation)
for i in range(len(constraintsAndPlasticities)):
    if (i == indBlank) | (np.isnan(constraintsAndPlasticities.iloc[i,13])):
        continue
    ax.text(initialSpacing+10*spacing,i-0.0,"{:.2e}".format(constraintsAndPlasticities.iloc[i,16]),fontsize=fontSize,color='k',horizontalalignment='left',verticalalignment='center')

# remove the x-axis ticks
ax.set_xticks([])

# make column "headers" by plotting ax.text above the table
# colHeaders = ['pattern','C','T','L','S','Ca','$\mathcal{M}$  ','$\mathcal{B}$  ','$\mathcal{R}$  ','$\mathcal{A}$  ','inside','outside','total','r','p','r$_{PIC}$','p$_{PIC}$']
colHeaders = ['C','T','L','S','Ca']
for i in range(len(colHeaders)):
    ax.text(i-0.0,-1,colHeaders[i],fontsize=fontSize,color='k',horizontalalignment='center',verticalalignment='center')
    
# plot the other headers
colHeaders2 = ['$\mathcal{M}$','$\mathcal{B}$','$\mathcal{R}$','$\mathcal{A}$']
for i in range(len(colHeaders2)):
    ax.text(initialSpacing+0.2+i*spacing,-1.00,colHeaders2[i],fontsize=fontSize,color='k',horizontalalignment='center',verticalalignment='center')
colHeaders3 = ['$\sigma_{c,B}^2$','$\sigma_{c,\\notin B}^2$','$\sigma_{B}^2$','r','p','r$_{PIC}$','p$_{PIC}$']
for i in range(len(colHeaders3)):
    ax.text(initialSpacing+12.5+i*spacing,-1.00,colHeaders3[i],fontsize=fontSize,color='k',horizontalalignment='center',verticalalignment='center')
    
# draw lines to distinguish between the constraints and plasticities
# constraints
# ax.annotate('', xy=(7.8, 0.245), xycoords='axes fraction', xytext=(7.8, 1),
# arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.annotate('', xy=(38.5, -0.5), xycoords='data', xytext=(38.5, numConstraints-0.5), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(39,numConstraints/2,'constraints',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# plasticities
ax.annotate('', xy=(38.5, numConstraints+0.5), xycoords='data', xytext=(38.5, numConstraints+numPlasticities+0.5), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(39,numConstraints+(numPlasticities/2)+0.5,'plasticities',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)

# now put the "type-I", "type-II", etc. labels on the left of the table
# do the same annotation brackets as the "constraints" and "plasticities" above but now on the left
# type-I
ax.annotate('', xy=(-5, -0.5), xycoords='data', xytext=(-5, numSingle-0.51), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(-5.5,(numSingle/2)-0.5,'I',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# type-II
ax.annotate('', xy=(-5, numSingle-0.49), xycoords='data', xytext=(-5, numSingle+numPluralAdjacent-0.51), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(-5.5,numSingle+(numPluralAdjacent/2)-0.5,'II',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# type-III
ax.annotate('', xy=(-5, numSingle+numPluralAdjacent-0.49), xycoords='data', xytext=(-5, numSingle+numPluralAdjacent+numPluralNonAdjacent-0.51), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(-5.5,numSingle+numPluralAdjacent+(numPluralNonAdjacent/2)-0.5,'III',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)
# "outsiders" or "other"
ax.annotate('', xy=(-5, numSingle+numPluralAdjacent+numPluralNonAdjacent-0.49), xycoords='data', xytext=(-5, numSingle+numPluralAdjacent+numPluralNonAdjacent+numPluralAdjacentOutsiders+numPluralNonAdjacentOutsiders-0.51), annotation_clip=False,
arrowprops=dict(arrowstyle="|-|", color='k', lw=1.0))
ax.text(-5.5,numSingle+numPluralAdjacent+numPluralNonAdjacent+(numPluralAdjacentOutsiders+numPluralNonAdjacentOutsiders)/2-0.5,'other',fontsize=fontSize,horizontalalignment='center',verticalalignment='center',rotation=90)

# save figure
plt.savefig(outputPath+'plots/constraintsPlasticitiesTableAll_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/constraintsPlasticitiesTableAll_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')

#%% determine the number of constraints as a function of the threshold

constraintsAllLowThreshold = pd.read_csv(inputPath+'additionalData/constraintMaster_lowThreshold.csv')

# we're using the thresholds of insideVariance/totalVariance and insideVariance/outsideVariance for the constraints
# we set these values at 0.1 and 0.1
thresholdi = ([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5])
# count the number of constraints found at each threshold
constraintsTest = constraintsAllLowThreshold.copy()
constraintsTest = constraintsTest[constraintsTest['insideVariance']!=0]
numConstraints = []
for i in range(len(thresholdi)):
    numConstraints.append(np.count_nonzero((constraintsTest['insideVariance']/constraintsTest['totalVariance'] < thresholdi[i]) | (constraintsTest['insideVariance']/constraintsTest['outsideVariance'] < thresholdi[i])))

# now for plasticities
threshold2i = ([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
numPlasticities = []
for i in range(len(threshold2i)):
    numPlasticities.append(np.count_nonzero((plasticitiesAll['insideVariance']/plasticitiesAll['totalVariance'] > threshold2i[i])))

#%% another large set of plots for the rest of the constraints and plasticities

# reduce the font size
fontSize = 12
markerSize = 5

# Create a figure and a gridspec layout
fig = plt.figure(figsize=(12, 5.5))#,constrained_layout=False)
fig.subplots_adjust(hspace=1.5,wspace=1.5)
gs = gridspec.GridSpec(4,8)#,hspace=30000000)

ax7 = plt.subplot(gs[0:2,0:2])
ax0 = plt.subplot(gs[0:2,2:4])
ax4 = plt.subplot(gs[0:2,4:6])
ax2 = plt.subplot(gs[0:2,6:8])
# ax3 = plt.subplot(gs[2:4,0:2]) # switch with ax5
ax5 = plt.subplot(gs[2:4,0:2])
ax1 = plt.subplot(gs[2:4,2:4])
# ax5 = plt.subplot(gs[2:4,4:6]) # switch with ax3
ax3 = plt.subplot(gs[2:4,4:6])
ax6 = plt.subplot(gs[2:4,6:8])

# plot the number of constraints and plasticities as a function of the threshold
ax7.plot(thresholdi,numConstraints,'o',color='k',markersize=markerSize,label='constraints')
ax7.plot(threshold2i,numPlasticities,'s',color='grey',markersize=markerSize,label='plasticities')
ax7.set_xlabel('threshold',fontsize=fontSize)
ax7.set_ylabel('number',fontsize=fontSize)
ax7.set_xlim([-0.1,1.1])
ax7.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax7.tick_params(labelsize=fontSize-1)

# plot vertical dashed lines at 0.1 and 0.8
ax7.axvline(x=0.1,linestyle='--',color='k',linewidth=1.0)
ax7.axvline(x=0.8,linestyle='--',color='grey',linewidth=1.0)

# legend on top right
ax7.legend(fontsize=fontSize-1,loc='center right',frameon=True,markerfirst=False,handletextpad=0.1)
ax7.text(-0.3,1.05,'A',fontsize=fontSize+4,fontweight='normal',transform=ax7.transAxes)

# 2T+Ca vs. L...just plot T instead of 2T?
# mammals
ax0.plot(vertebralData['Lumbar'],vertebralData['Thoracic']+vertebralData['Caudal'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label=r'$\mathcal{M}$ (Mammalia)')
# birds
ax0.plot(birds['Lumbar'],birds['Thoracic']+birds['Caudal'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label=r'$\mathcal{B}$ (Aves)')
# reptiles
ax0.plot(reptiles['Lumbar'],reptiles['Thoracic']+reptiles['Caudal'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label=r'$\mathcal{R}$ (Reptilia)')
# amphibians
ax0.plot(amphibians['Lumbar'],amphibians['Thoracic']+amphibians['Caudal'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label=r'$\mathcal{A}$ (Amphibia)')

ax0.set_xlabel('Lumbar',fontsize=fontSize)
ax0.set_ylabel('Thoracic+Caudal',fontsize=fontSize)
ax0.annotate(r'$\mathcal{A}$ (Amphibia)',
    xy=(0.9,0.85), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[3],ha='right')
ax0.annotate(r'$\mathcal{M}$ (Mammalia)',
    xy=(0.9,0.70), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right')
ax0.annotate(r'$\mathcal{B}$ (Aves)',
    xy=(0.9,0.55), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[1],ha='right')
ax0.annotate(r'$\mathcal{R}$ (Reptilia)',
    xy=(0.9,0.40), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[2],ha='right')

ax0.text(-0.3,1.05,'B',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)

# C-T-S-Ca: load the species from node 1270 and plot T+S+Ca vs. C
species1270 = pd.read_csv(outputPath+'vertebral/subTree_1270_1270_speciesNum36/species.csv',header=None)[0].to_list()
# only plot these species (reptiles)
r,p = scipy.stats.pearsonr(vertebralData['Cervical'][vertebralData['species'].isin(species1270)],vertebralData['Thoracic'][vertebralData['species'].isin(species1270)]+vertebralData['Sacral'][vertebralData['species'].isin(species1270)]+vertebralData['Caudal'][vertebralData['species'].isin(species1270)])
ax1.plot(birds['Cervical'][birds['species'].isin(species1270)],birds['Thoracic'][birds['species'].isin(species1270)]+birds['Sacral'][birds['species'].isin(species1270)]+birds['Caudal'][birds['species'].isin(species1270)],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax1.set_xlabel('Cervical',fontsize=fontSize)
ax1.set_ylabel('Thoracic+Sacral+Caudal',fontsize=fontSize)
# ax1.set_title('r='+str(round(r,2))+'***, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1270')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==-1)].iloc[0],2))+'**',fontsize=fontSize)
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1270')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==-1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='1270')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==-1)].iloc[0]
ax1.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax1.set_xlim([10,25])
ax1.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
ax1.text(0.95,0.05,'Galloanserae',transform=ax1.transAxes,fontsize=fontSize-1,ha='right')
ax1.text(-0.3,1.05,'F',fontsize=fontSize+4,fontweight='normal',transform=ax1.transAxes)
birds['Thoracic+Sacral+Caudal'] = birds['Thoracic']+birds['Sacral']+birds['Caudal']
saveForPIC(species1270,'Cervical','Thoracic+Sacral+Caudal',birds,tree,'_node1270')

# C-S-Ca: load the species from node 1208 and plot S+Ca vs. C
species1208 = pd.read_csv(outputPath+'vertebral/subTree_1208_1208_speciesNum20/species.csv',header=None)[0].to_list()
# only plot these species (reptiles)
r,p = scipy.stats.pearsonr(vertebralData['Cervical'][vertebralData['species'].isin(species1208)],vertebralData['Sacral'][vertebralData['species'].isin(species1208)]+vertebralData['Caudal'][vertebralData['species'].isin(species1208)])
ax2.plot(birds['Cervical'][birds['species'].isin(species1208)],birds['Sacral'][birds['species'].isin(species1208)]+birds['Caudal'][birds['species'].isin(species1208)],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax2.set_xlabel('Cervical',fontsize=fontSize)
ax2.set_ylabel('Sacral+Caudal',fontsize=fontSize)
# ax2.set_title('r='+str(round(r,2))+'***, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1208')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['S']==-1)].iloc[0],2))+'***',fontsize=fontSize)
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1208')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['S']==-1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='1208')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['S']==-1)].iloc[0]
ax2.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax2.set_xlim([10,25])
ax2.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
ax2.text(0.95,0.05,'Anseriformes',transform=ax2.transAxes,fontsize=fontSize-1,ha='right')
ax2.text(-0.3,1.05,'D',fontsize=fontSize+4,fontweight='normal',transform=ax2.transAxes)
birds['Sacral+Caudal'] = birds['Sacral']+birds['Caudal']
saveForPIC(species1208,'Cervical','Sacral+Caudal',birds,tree,'_node1208')


# # next
# species1351 = pd.read_csv(outputPath+'vertebral/subTree_1351_1351_speciesNum146/species.csv',header=None)[0].to_list()
# r,p = scipy.stats.pearsonr(vertebralData['Thoracic'][vertebralData['species'].isin(species1351)],vertebralData['Sacral'][vertebralData['species'].isin(species1351)]+vertebralData['Caudal'][vertebralData['species'].isin(species1351)])
# ax3.plot(birds['Thoracic'][birds['species'].isin(species1351)],birds['Sacral'][birds['species'].isin(species1351)]+birds['Caudal'][birds['species'].isin(species1351)],'s',markersize=markerSize+1,color=colorWheel[1],alpha=0.5,label='Birds')
# ax3.plot(reptiles['Thoracic'][reptiles['species'].isin(species1351)],reptiles['Sacral'][reptiles['species'].isin(species1351)]+reptiles['Caudal'][reptiles['species'].isin(species1351)],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label='Reptiles')
# ax3.set_xlabel('Thoracic',fontsize=fontSize)
# ax3.set_ylabel('Sacral+Caudal',fontsize=fontSize)
# ax3.set_title('r='+str(round(r,2))+'***, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['S']==-1)].iloc[0],2))+'***',fontsize=fontSize)
# rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['S']==-1)].iloc[0]
# pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='1351')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['S']==-1)].iloc[0]
# ax3.set_xlim([2,25])
# ax3.set_xticks([5,10,15,20,25])
# ax3.set_ylim([10,45])
# ax3.set_yticks([10,20,30,40])
# ax3.tick_params(labelsize=fontSize-1)
# print(r,p,rPIC,pPIC)
# # fit to all in species1351
# m = np.polyfit(vertebralData['Cervical'][vertebralData['species'].isin(species1351)],vertebralData['Thoracic'][vertebralData['species'].isin(species1351)],1)
# xi = np.linspace(np.nanmin(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),np.nanmax(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),100)
# ax3.text(0.95,0.05,'Testudinata,\nCrocodilia,\nAves',transform=ax3.transAxes,fontsize=fontSize-1,ha='right')
# ax3.text(-0.3,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax3.transAxes)

# next
species151 = pd.read_csv(outputPath+'vertebral/subTree_151_151_speciesNum73/species.csv',header=None)[0].to_list()
r,p = scipy.stats.pearsonr(vertebralData['Thoracic'][vertebralData['species'].isin(species151)],vertebralData['Caudal'][vertebralData['species'].isin(species151)])
ax3.plot(amphibians['Thoracic'][amphibians['species'].isin(species151)],amphibians['Caudal'][amphibians['species'].isin(species151)],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label='Reptiles')
ax3.set_xlabel('Thoracic',fontsize=fontSize)
ax3.set_ylabel('Caudal',fontsize=fontSize)
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='151')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['Ca']==-1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='151')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['Ca']==-1)].iloc[0]
ax3.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)

# ax3.set_xlim([2,25])
ax3.set_xticks([0,10,20,30,40,50,60])
# ax3.set_ylim([10,45])
ax3.set_yticks([0,10,20,30,40])
ax3.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
# fit to all in species151
# xi = np.linspace(np.nanmin(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),np.nanmax(vertebralData['Cervical'][vertebralData['species'].isin(species1351)]),100)
ax3.text(0.95,0.05,'Anura,\nUrodela',transform=ax3.transAxes,fontsize=fontSize-1,ha='right')
ax3.text(-0.3,1.05,'G',fontsize=fontSize+4,fontweight='normal',transform=ax3.transAxes)
saveForPIC(species151,'Thoracic','Caudal',amphibians,tree,'_node151')

# circle the outlier: Amphiuma means
additionalCircleSize = 6
ax3.plot(amphibians['Thoracic'][amphibians['species']=='Amphiuma means']-0.25,amphibians['Caudal'][amphibians['species']=='Amphiuma means'],marker='o',color='black',linestyle='None',mfc='None',alpha=0.5,markersize=markerSize+additionalCircleSize)
ax3.text(amphibians['Thoracic'][amphibians['species']=='Amphiuma means'],amphibians['Caudal'][amphibians['species']=='Amphiuma means']-1.5,'$\it{Amphiuma}$\n$\it{means}$',fontsize=fontSize-2,ha='right',va='top')


# T-2Ca vs. S...just plot Ca instead of 2Ca?
# mammals
ax4.plot(vertebralData['Sacral'],vertebralData['Thoracic']*1+vertebralData['Caudal'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label=r'$\mathcal{M}$ (Mammalia)')
# birds
ax4.plot(birds['Sacral'],birds['Thoracic']*1+birds['Caudal'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label=r'$\mathcal{B}$ (Aves)')
# reptiles
ax4.plot(reptiles['Sacral'],reptiles['Thoracic']*1+reptiles['Caudal'],'^',markersize=markerSize,color=colorWheel[2],alpha=0.5,label=r'$\mathcal{R}$ (Reptilia)')
# amphibians
ax4.plot(amphibians['Sacral'],amphibians['Thoracic']*1+amphibians['Caudal'],'>',markersize=markerSize,color=colorWheel[3],alpha=0.5,label=r'$\mathcal{A}$ (Amphibia)')

ax4.set_xlabel('Sacral',fontsize=fontSize)
ax4.set_ylabel('Thoracic+Caudal',fontsize=fontSize)

ax4.text(-0.3,1.05,'C',fontsize=fontSize+4,fontweight='normal',transform=ax4.transAxes)

# make the significance block
# add inset on top right that extends outside the figure axis on the top and the right
axins = ax4.inset_axes([0.5,0.55,0.55,0.5])
# remove all ticks
axins.set_xticks([])
axins.set_yticks([])
# annotate:
# "p-value
# < 0.05 *
# < 0.01 **
# < 0.001 ***"
axins.annotate('p-value\n< 0.05 *\n< 0.01 **\n< 0.001 ***',
    xy=(0.08,0.88), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',ha='left',va='top',fontsize=fontSize)

# # C+T-Ca: load the species from node 945 and plot Ca vs. C+T
# species945 = pd.read_csv(outputPath+'vertebral/subTree_945_945_speciesNum30/species.csv',header=None)[0].to_list()
# # only plot these species (reptiles)
# r,p = scipy.stats.pearsonr(vertebralData['Cervical'][vertebralData['species'].isin(species945)]+vertebralData['Thoracic'][vertebralData['species'].isin(species945)],vertebralData['Caudal'][vertebralData['species'].isin(species945)])
# ax5.plot(birds['Cervical'][birds['species'].isin(species945)]+birds['Thoracic'][birds['species'].isin(species945)],birds['Caudal'][birds['species'].isin(species945)],'s',markersize=markerSize+1,color=colorWheel[1],alpha=0.5,label='Birds')
# ax5.set_xlabel('Cervical',fontsize=fontSize)
# ax5.set_ylabel('Sacral+Caudal',fontsize=fontSize)
# # ax5.set_title('r='+str(round(r,2))+'*, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='945')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0],2))+'*',fontsize=fontSize)
# rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='945')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0]
# pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='945')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0]
# ax5.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# ax5.set_xlim([15,20])
# ax5.set_ylim([5,10])
# ax5.tick_params(labelsize=fontSize-1)
# print(r,p,rPIC,pPIC)
# ax5.text(0.46,0.88,'Telluraves',transform=ax5.transAxes,fontsize=fontSize-1,ha='right')
# ax5.text(-0.3,1.05,'G',fontsize=fontSize+4,fontweight='normal',transform=ax5.transAxes)
# birds['Cervical+Thoracic'] = birds['Cervical']+birds['Thoracic']
# saveForPIC(species945,'Cervical+Thoracic','Caudal',birds,tree,'_node945')

# # T-Ca: load the species from node 943 and plot Ca vs. T
species943 = pd.read_csv(outputPath+'vertebral/subTree_943_943_speciesNum78/species.csv',header=None)[0].to_list()
# only plot these species (reptiles)
r,p = scipy.stats.pearsonr(vertebralData['Thoracic'][vertebralData['species'].isin(species943)],vertebralData['Caudal'][vertebralData['species'].isin(species943)])
# r,p = scipy.stats.pearsonr(birds['Thoracic'],birds['Caudal'])
ax5.plot(birds['Thoracic'][birds['species'].isin(species943)],birds['Caudal'][birds['species'].isin(species943)],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
# ax5.plot(birds['Thoracic'],birds['Caudal'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax5.set_xlabel('Thoracic',fontsize=fontSize)
ax5.set_ylabel('Caudal',fontsize=fontSize)
# ax5.set_title('r='+str(round(r,2))+'*, PIC: r='+str(round(constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='945')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['T']==1)].iloc[0],2))+'*',fontsize=fontSize)
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='943')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['Ca']==-1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='943')&(constraintsSortedByNode['T']==1)&(constraintsSortedByNode['Ca']==-1)].iloc[0]
# rPIC,pPIC = scipy.stats.pearsonr(birdsPIC['T'],birdsPIC['Ca'])
ax5.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
ax5.set_xlim([2,10])
ax5.set_xticks([2,4,6,8,10])
ax5.set_ylim([2,10])
ax5.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
ax5.text(0.95,0.05,'Neognathae less\nGalloanserae,\nAccipitriformes,\nCariamiformes',transform=ax5.transAxes,fontsize=fontSize-1,ha='right')
ax5.text(-0.3,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax5.transAxes)
saveForPIC(species943,'Thoracic','Caudal',birds,tree,'_node943')

# constraint: C-L-S = 0 (so plot L+S vs. C)
# ax0 = plt.subplot(gs[0,0])
r,p = scipy.stats.pearsonr(vertebralData['Cervical'],vertebralData['Lumbar']+vertebralData['Sacral'])
rPIC = constraintsSortedByNode['rPIC'][(constraintsSortedByNode['node']=='full')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['L']==-1)].iloc[0]
pPIC = constraintsSortedByNode['pPIC'][(constraintsSortedByNode['node']=='full')&(constraintsSortedByNode['C']==1)&(constraintsSortedByNode['L']==-1)].iloc[0]
# ax0.plot(vertebralData['Cervical'],vertebralData['Sacral'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax6.plot(mammals['Cervical'],mammals['Lumbar']+mammals['Sacral'],'o',markersize=markerSize,color=colorWheel[0],alpha=0.5,label='Mammals')
ax6.plot(birds['Cervical'],birds['Lumbar']+birds['Sacral'],'s',markersize=markerSize,color=colorWheel[1],alpha=0.5,label='Birds')
ax6.plot(reptiles['Cervical'],reptiles['Lumbar']+reptiles['Sacral'],'^',markersize=markerSize+1,color=colorWheel[2],alpha=0.5,label='Reptiles')
ax6.plot(amphibians['Cervical'],amphibians['Lumbar']+amphibians['Sacral'],'>',markersize=markerSize+1,color=colorWheel[3],alpha=0.5,label='Amphibians')
ax6.set_xlabel('Cervical',fontsize=fontSize)
ax6.set_ylabel('Lumbar+Sacral',fontsize=fontSize)
ax6.set_title('r='+str(round(r,2))+getAsterisks(p)+', PIC: r='+str(round(rPIC,2))+getAsterisks(pPIC),fontsize=fontSize)
# plot C=S line
ax6.plot([0,25],[0,25],'k--',linewidth=lineWidth)
# add text for C=S
ax6.text(0.45,0.82,'C=L+S',transform=ax6.transAxes,fontsize=fontSize)
ax6.set_xticks([0,5,10,15,20,25])
ax6.set_yticks([0,5,10,15,20,25])
# make tick labels smaller
ax6.tick_params(labelsize=fontSize-1)
print(r,p,rPIC,pPIC)
vertebralData['Lumbar+Sacral'] = vertebralData['Lumbar']+vertebralData['Sacral']
saveForPIC(vertebralData['species'].to_list(),'Cervical','Lumbar+Sacral',vertebralData,tree,'_full')

ax6.text(-0.3,1.05,'H',fontsize=fontSize+4,fontweight='normal',transform=ax6.transAxes)

plt.savefig(outputPath+'plots/constraintsPlasticitiesPlotsAll_extendedDataFigure_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/constraintsPlasticitiesPlotsAll_extendedDataFigure_v2.pdf',dpi=300,bbox_inches='tight')
