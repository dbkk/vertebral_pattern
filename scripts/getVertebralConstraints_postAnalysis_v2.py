# python script to gather all the PCA and human-readable results together
# v2

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath
outputPath = inputPath

#%% import libraries

from skbio import TreeNode
from io import StringIO
import pandas as pd
import datetime
import numpy as np
import os
import sympy as sp
import copy
import scipy.cluster.hierarchy as spc
import glob

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
faceColor = 'aliceblue'
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


#%% function for making the PCA constraints human readable

def humanReadableConstraintMatrix(spanHumanReadable,toCheck):
    
    # making the 'interpretable' constraint vectors
    const_vec = np.array([np.arange(-spanHumanReadable,spanHumanReadable+1)]*len(toCheck))
    cvec = np.array(np.meshgrid(*const_vec)).T.reshape(-1,len(toCheck))
    cvec = cvec[np.sum(np.abs(cvec),axis=1)!=0,:]
    cvec_unique=[]
    for i in range(cvec.shape[0]):
        if (tuple(cvec[i,:]) not in cvec_unique) & (tuple(-cvec[i,:]) not in cvec_unique):
            cvec_unique.append(tuple(cvec[i,:]))
    cvec_unique=np.array(cvec_unique)
    
    return cvec_unique

#%% function for defining the correlation matrix

def corrMatrixFunction(constraints,startCol,endCol):
    # the pandas native way gives nans for some reason...
    corrMat = np.zeros((len(constraints),len(constraints)))
    for i in range(len(constraints)):
        for j in range(len(constraints)):
            corrMat[i,j] = np.inner(constraints.iloc[i,startCol:endCol].values,constraints.iloc[j,startCol:endCol].values)/(np.sqrt(np.inner(constraints.iloc[i,startCol:endCol].values,constraints.iloc[i,startCol:endCol].values))*np.sqrt(np.inner(constraints.iloc[j,startCol:endCol].values,constraints.iloc[j,startCol:endCol].values)))
    return corrMat

#%% simple function to simplify constraints

def simplifyConstraints(x):
    
    # change the sign of the constraint vectors so that the largest non-zero element is positive
    for i in range(len(x)):
        indLargest = np.argmax(np.abs(x[i,:]))
        if x[i,indLargest]<0:
            x[i,:] = -x[i,:]
    
    return x

#%% another simplifier

def simplifyConstraints2(x):

    # change the sign of the constraint vectors so that the first non-zero element is positive
    for i in range(len(x)):
        indNonZero = np.where(x[i,:]!=0)[0][0]
        if x[i,indNonZero]<0:
            x[i,:] = -x[i,:]
            
    # make sure we reduce/simplify the constraint vectors as much as possible
    for i in range(len(x)):
        x[i,:] = x[i,:]/np.min(np.abs(x[i,x[i,:]!=0]))
    
    x += 0.
    
    return x

#%% Define a function to check if one node is a descendant of another (suggested by ChatGPT)

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


#%% load from a file or use these default values

# the parameter file is the only one with a .txt extension
parameterFile = glob.glob(inputPath+'*.txt')

if os.path.exists(parameterFile[0]):
    print('Loading parameters from file...')
    f = open(parameterFile[0],"r")
    linesAll = f.readlines()
    f.close()
    f = open(parameterFile[0],"r")
    cnt = 0
    for line in f:
        if line.startswith('numThresh'):
            numThresh = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('constraintRatio '):
            constraintRatio = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('constraintRatioHumanReadable '):
            constraintRatioHumanReadable = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('constraintThresholdLocal '):
            constraintThresholdLocal = float(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('plasticityThresholdLocal '):
            plasticityThresholdLocal = float(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('spanHumanReadable'):
            spanHumanReadable = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('plasticityRatio'):
            plasticityRatio = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('constraintName'):
            constraintName = str(line.split(' ')[1].strip('\n'))
            cnt = cnt + 1
        elif line.startswith('fillNaNs'):
            fillNaNs = bool(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('maxPCA'):
            maxPCA = int(line.split(' ')[1])
            cnt = cnt + 1
        elif line.startswith('subgroup'):
            subgroup = str(line.split(' ')[1].strip('\n'))
            cnt = cnt + 1
        elif line.startswith('featureList'):
            cnt = cnt + 1
    toCheck = []
    for i in range(cnt,len(linesAll)):
        toCheck.append(linesAll[i].rstrip('\n'))
    f.close()
else:
    print('Using default parameters...')
    numThresh = 10
    constraintRatio = 100
    constraintRatioHumanReadable = 10
    spanHumanReadable = 3
    plasticityRatio = 5
    constraintName = 'vertebral'
    fillNaNs = True # if false we remove those species
    maxPCA = 10
    subgroup = 'Class'
    toCheck = (['Cervical','Thoracic','Lumbar','Sacral','Caudal'])

if constraintName == 'vertebral':
    dataCols = ['C','T','L','S','Ca']
else:
    dataCols = toCheck
    

#%% load the tree

treePath = outputPath+constraintName+'/fullFormulaTree/tree.nwk'
with open(treePath) as f:
    treeFile = f.read()
tree = TreeNode.read(StringIO(treeFile))

# I actually want my own ordering here for my color scheme:
groupList = (['Mammalia','Aves','Reptilia','Amphibia'])

#%% get the full vertebral data

vertebralData = pd.read_csv(outputPath+constraintName+'/fullFormulaTree/vertebralData.csv')

#%% get a list of the nodes from the directories in the base directory

files = os.listdir(outputPath+constraintName+'/')
# get all the nodes which go as "subTree_NODE1_NODE2_speciesNumSPECIESNUMBER"
node1List = []
node2List = []
speciesNumList = []
# first add the "root" node for which the results are in the directory "fullFormulaTree"
node1List.append('full')
node2List.append('full')
speciesNumList.append(len(pd.read_csv(outputPath+constraintName+'/fullFormulaTree/species.csv',header=None)))
for i in range(len(files)):
    if files[i].split('_')[0] == 'subTree':
        node1List.append(files[i].split('_')[1])
        node2List.append(files[i].split('_')[2])
        speciesNumList.append(files[i].split('_')[3].split('.')[0][len('speciesNum'):])
        
#%% first concatenate the PCA results

nodes = []
loadings = []
explainedVariances = []
explainedVariancePercentages = []
numMammals = []
numBirds = []
numReptiles = []
numAmphibians = []
for i in range(len(node1List)):
    if node1List[i] == node2List[i]:
        if node1List[i] == 'full':
            dirName = 'fullFormulaTree'
        else:
            dirName = 'subTree_'+node1List[i]+'_'+node2List[i]+'_speciesNum'+speciesNumList[i]
        loadingsTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/loadings.csv',delimiter=',')
        explainedVarianceTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/explainedVariance.csv',delimiter=',')
        explainedVariancePercentageTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/explainedPercentage.csv',delimiter=',')
        # load the class data which is a list of strings
        classTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/class.csv',delimiter=',',dtype=str)
        numMammalsTemp = np.sum(classTemp=='Mammalia')
        numBirdsTemp = np.sum(classTemp=='Aves')
        numReptilesTemp = np.sum(classTemp=='Reptilia')
        numAmphibiansTemp = np.sum(classTemp=='Amphibia')
        for j in range(len(loadingsTemp)):
            nodes.append(node1List[i])
            loadings.append(loadingsTemp[j])
            explainedVariances.append(explainedVarianceTemp[j])
            explainedVariancePercentages.append(explainedVariancePercentageTemp[j])
            numMammals.append(numMammalsTemp)
            numBirds.append(numBirdsTemp)
            numReptiles.append(numReptilesTemp)
            numAmphibians.append(numAmphibiansTemp)
            
# make an array with number of rows equal to loadings and number of columns equal to dataCols
loadingsArray = np.zeros((len(loadings),len(dataCols)))
for i in range(len(loadings)):
    loadingsArray[i,:] = loadings[i]
    
# simplify
loadingsArray = simplifyConstraints(loadingsArray)
    
# make a dataframe of the loadings and explained variances
pca = pd.DataFrame({'node':nodes,'explainedVariance':explainedVariances,'explainedVariancePercentage':explainedVariancePercentages,'numMammals':numMammals,'numBirds':numBirds,'numReptiles':numReptiles,'numAmphibians':numAmphibians})
# add in the dataCols after 'node' and before 'explainedVariances'
for i in range(len(dataCols)):
    pca.insert(i+1,dataCols[i],loadingsArray[:,i])
    
#%% find the actual inside and outside variance for each node
# the previous ones were normalized differently

outsideVariance = []
insideVariance = []
totalVariance = []
explainedVariance = []
for i in range(len(pca)):
    node = pca['node'].iloc[i]
    if node == 'full':
        outsideVariance.append(np.nan)
        insideVariance.append(np.var(np.dot(pca.iloc[i,1:6].values,vertebralData.iloc[:,1:6].values.T)))
        totalVariance.append(np.sum(np.var(vertebralData.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
    else:
        # get the species in this node
        species = pd.read_csv(outputPath+constraintName+'/subTree_'+node+'_'+node+'_speciesNum'+str(pca['numMammals'].iloc[i]+pca['numBirds'].iloc[i]+pca['numReptiles'].iloc[i]+pca['numAmphibians'].iloc[i])+'/species.csv',header=None)[0].to_list()
        inside = vertebralData[vertebralData['species'].isin(species)]
        # dot product the pca loading by the data values and calculate the variance
        insideVariance.append(np.var(np.dot(pca.iloc[i,1:6].values,inside.iloc[:,1:6].values.T)))
        totalVariance.append(np.sum(np.var(inside.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
        outside = vertebralData[~vertebralData['species'].isin(species)]
        outsideVariance.append(np.var(np.dot(pca.iloc[i,1:6].values,outside.iloc[:,1:6].values.T)))  
# rename the previous insideVariance and explainedVariance 
pca.rename(columns={'explainedVariance':'explainedVarianceOld','explainedVariancePercentage':'explainedVariancePercentageOld'},inplace=True)
# add in the new ones after the 'Ca' column
# avoid the TypeError: loc must be int
indCol = int(np.where(pca.columns=='Ca')[0][0])
pca.insert(indCol+1,'insideVariance',np.array(insideVariance))
pca.insert(indCol+2,'totalVariance',np.array(totalVariance))
pca.insert(indCol+3,'explainedVariance',np.array(explainedVariance))
pca.insert(indCol+4,'outsideVariance',np.array(outsideVariance))
        
#%% consider all those loadings which have a variance explained less than the threshold constraintThresholdLocal

pcaConstraintLocal = pca[(pca['explainedVariance'] < constraintThresholdLocal) | (pca['insideVariance']/pca['outsideVariance'] < 1/constraintRatio)]
# pcaConstraintLocal = pca[pca['explainedVariancePercentage'] < constraintThresholdLocal]
# reset index
pcaConstraintLocal.reset_index(inplace=True,drop=True)

#%% reorganize the dataframe so that we have a column for each node and a row for each species

methodName = 'complete'
# methodName = 'single'

corrMat = corrMatrixFunction(pcaConstraintLocal,1,6)

# the pandas native way gives nans for some reason...
pdist = spc.distance.pdist(corrMat)
linkage = spc.linkage(pdist, method=methodName)
idx = spc.fcluster(linkage, 0.25 * pdist.mean(), 'distance')
# add the cluster to the dataframe
pcaConstraintLocal['cluster'] = idx
# determine which cluster has the most constituents
unique, counts = np.unique(idx, return_counts=True)
# rearrange the tempMammal dataframe by these clusters starting from the most populous cluster
newIndices = []
countThresh = 3
indSort = np.argsort(counts)[::-1]
unique = unique[indSort]
counts = counts[indSort]
for i in range(len(unique)):
    if counts[i] >= countThresh:
        temp = np.where(idx==unique[i])[0]
        for j in range(len(temp)):
            newIndices.append(temp[j])
pcaConstraintLocal_organizedTemp = pcaConstraintLocal.iloc[newIndices,:]
pcaConstraintLocal_organized = pcaConstraintLocal_organizedTemp

corrMat_organized = corrMatrixFunction(pcaConstraintLocal_organized,1,6)

#%% make human-readable version of each of the big groups of constraints

# first taking their average value
clusters = pcaConstraintLocal_organized['cluster'].unique()
meanConstraints = np.zeros((len(clusters),len(dataCols)))
for i in range(len(clusters)):
    meanConstraints[i,:] = np.mean(pcaConstraintLocal_organized[pcaConstraintLocal_organized['cluster']==clusters[i]].iloc[:,1:6].values,axis=0)
    
# make human-readable versions of each cluster constraints:
cvec_unique = humanReadableConstraintMatrix(3,dataCols)

readableConstraints = np.zeros((len(meanConstraints),len(dataCols)))
for i in range(len(meanConstraints)):
    readableConstraints[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,meanConstraints[i,:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
    
readableConstraints = simplifyConstraints2(readableConstraints)

# make an array with number of rows equal to the current dataframe
# and number of columns equal to dataCols
# and fill it with the human-readable constraints corresponding to the clusters in the dataframe
readableConstraintsArray = np.zeros((len(pcaConstraintLocal_organized),len(dataCols)))
for i in range(len(clusters)):
    readableConstraintsArray[pcaConstraintLocal_organized['cluster']==clusters[i],:] = readableConstraints[i,:]

# add this to the dataframe
for i in range(len(dataCols)):
    pcaConstraintLocal_organized.insert(i+1,dataCols[i]+'_readable',readableConstraintsArray[:,i])


#%% # now do the human-readable : concatenate everything, keeping track of the nodes

# for now just do the nodes which are the same

if constraintName == 'vertebral':
    dataCols = ['C','T','L','S','Ca']
else:
    dataCols = toCheck

nodes = []
nodesB = []
localVariance = []
outsideVariance = []
totalVariance = []
cervical = []
thoracic = []
lumbar = []
sacral = []
caudal = []
numMammals = []
numBirds = []
numReptiles = []
numAmphibians = []
for i in range(len(node1List)):
    if node1List[i] == node2List[i]:
        if node1List[i] == 'full':
            dirName = 'fullFormulaTree'
        else:
            dirName = 'subTree_'+node1List[i]+'_'+node2List[i]+'_speciesNum'+speciesNumList[i]
        constraints = pd.read_csv(outputPath+constraintName+'/'+dirName+'/constraints_humanReadable.csv')
        # load the class data which is a list of strings
        classTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/class.csv',delimiter=',',dtype=str)
        numMammalsTemp = np.sum(classTemp=='Mammalia')
        numBirdsTemp = np.sum(classTemp=='Aves')
        numReptilesTemp = np.sum(classTemp=='Reptilia')
        numAmphibiansTemp = np.sum(classTemp=='Amphibia')
        for j in range(len(constraints)):
            nodes.append(node1List[i])
            localVariance.append(constraints['std'].iloc[j]**2)
            outsideVariance.append(constraints['outside_std'].iloc[j]**2)
            totalVariance.append(constraints['totalVar'].iloc[j])
            numMammals.append(numMammalsTemp)
            numBirds.append(numBirdsTemp)
            numReptiles.append(numReptilesTemp)
            numAmphibians.append(numAmphibiansTemp)
            cervical.append(constraints['Cervical'].iloc[j])
            thoracic.append(constraints['Thoracic'].iloc[j])
            lumbar.append(constraints['Lumbar'].iloc[j])
            sacral.append(constraints['Sacral'].iloc[j])
            caudal.append(constraints['Caudal'].iloc[j])

explainedVariancePercentage = np.array(np.array(localVariance)/np.array(totalVariance))
            
# make a dataframe of the loadings and explained variances
hr = pd.DataFrame({'node':nodes,'C':cervical,'T':thoracic,'L':lumbar,'S':sacral,'Ca':caudal,'localVariance':localVariance,'outsideVariance':outsideVariance,'totalVariance':totalVariance,'explainedVariancePercentage':explainedVariancePercentage,'numMammals':numMammals,'numBirds':numBirds,'numReptiles':numReptiles,'numAmphibians':numAmphibians})

#%% find the actual inside and outside variance for each node
# the previous ones were normalized differently

outsideVariance = []
insideVariance = []
totalVariance = []
explainedVariance = []
for i in range(len(hr)):
    node = hr['node'].iloc[i]
    if node == 'full':
        outsideVariance.append(np.nan)
        insideVariance.append(np.var(np.dot(hr.iloc[i,1:6].values,vertebralData.iloc[:,1:6].values.T)))
        totalVariance.append(np.sum(np.var(vertebralData.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
    else:
        # get the species in this node
        species = pd.read_csv(outputPath+constraintName+'/subTree_'+node+'_'+node+'_speciesNum'+str(hr['numMammals'].iloc[i]+hr['numBirds'].iloc[i]+hr['numReptiles'].iloc[i]+hr['numAmphibians'].iloc[i])+'/species.csv',header=None)[0].to_list()
        inside = vertebralData[vertebralData['species'].isin(species)]
        # dot product the pca loading by the data values and calculate the variance
        insideVariance.append(np.var(np.dot(hr.iloc[i,1:6].values,inside.iloc[:,1:6].values.T))/np.sum(hr.iloc[i,1:6].values**2))
        totalVariance.append(np.sum(np.var(inside.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
        outside = vertebralData[~vertebralData['species'].isin(species)]
        outsideVariance.append(np.var(np.dot(hr.iloc[i,1:6].values,outside.iloc[:,1:6].values.T))/np.sum(hr.iloc[i,1:6].values**2))  
# rename the previous insideVariance and explainedVariance 
hr.rename(columns={'localVariance':'localVarianceOld','outsideVariance':'outsideVarianceOld','totalVariance':'totalVarianceOld','explainedVariancePercentage':'explainedVariancePercentageOld'},inplace=True)
# add in the new ones after the 'Ca' column
# avoid the TypeError: loc must be int
indCol = int(np.where(pca.columns=='Ca')[0][0])
hr.insert(indCol+1,'insideVariance',np.array(insideVariance))
hr.insert(indCol+2,'totalVariance',np.array(totalVariance))
hr.insert(indCol+3,'explainedVariance',np.array(explainedVariance))
hr.insert(indCol+4,'outsideVariance',np.array(outsideVariance))


#%% filter out the constraints

hr = hr[(hr['explainedVariance'] < constraintThresholdLocal) | (hr['insideVariance']/hr['outsideVariance'] < 1/constraintRatio)]

#%% make a combined dataframe with the human-readable PCA (pcaConstraintLocal_organized) and the human-readable constraints (hr)
# keep the following shared information:
# node, explainedVariancePercentage, numMammals, numBirds, numReptiles, numAmphibians, 
# C (_readable for pca), T (_readable for pca), L (_readable for pca), S (_readable for pca), Ca (_readable for pca)

constraintMaster = pd.DataFrame()
# combine each of the shared columns from both dataframes
constraintMaster['node'] = pd.concat([pcaConstraintLocal_organized['node'],hr['node']],axis=0)
constraintMaster['C'] = pd.concat([pcaConstraintLocal_organized['C_readable'],hr['C']],axis=0)
constraintMaster['T'] = pd.concat([pcaConstraintLocal_organized['T_readable'],hr['T']],axis=0)
constraintMaster['L'] = pd.concat([pcaConstraintLocal_organized['L_readable'],hr['L']],axis=0)
constraintMaster['S'] = pd.concat([pcaConstraintLocal_organized['S_readable'],hr['S']],axis=0)
constraintMaster['Ca'] = pd.concat([pcaConstraintLocal_organized['Ca_readable'],hr['Ca']],axis=0)
constraintMaster['insideVariance'] = pd.concat([pcaConstraintLocal_organized['insideVariance'],hr['insideVariance']],axis=0)
constraintMaster['totalVariance'] = pd.concat([pcaConstraintLocal_organized['totalVariance'],hr['totalVariance']],axis=0)
constraintMaster['explainedVariance'] = pd.concat([pcaConstraintLocal_organized['explainedVariance'],hr['explainedVariance']],axis=0)
constraintMaster['outsideVariance'] = pd.concat([pcaConstraintLocal_organized['outsideVariance'],hr['outsideVariance']],axis=0)
constraintMaster['numMammals'] = pd.concat([pcaConstraintLocal_organized['numMammals'],hr['numMammals']],axis=0)
constraintMaster['numBirds'] = pd.concat([pcaConstraintLocal_organized['numBirds'],hr['numBirds']],axis=0)
constraintMaster['numReptiles'] = pd.concat([pcaConstraintLocal_organized['numReptiles'],hr['numReptiles']],axis=0)
constraintMaster['numAmphibians'] = pd.concat([pcaConstraintLocal_organized['numAmphibians'],hr['numAmphibians']],axis=0)

# reset the index
constraintMaster.reset_index(inplace=True,drop=True)

# now go through each constraint, and get the list of species corresponding to that constraint's node
# and add that list to the constraintMaster dataframe
speciesList = []
for i in range(len(constraintMaster)):
    ind = node1List.index(constraintMaster['node'].iloc[i])
    if constraintMaster['node'].iloc[i] == 'full':
        speciesList.append(np.loadtxt(outputPath+constraintName+'/fullFormulaTree/species.csv',delimiter=',',dtype=str))
    else:
        speciesList.append(np.loadtxt(outputPath+constraintName+'/subTree_'+node1List[ind]+'_'+node2List[ind]+'_speciesNum'+speciesNumList[ind]+'/species.csv',delimiter=',',dtype=str))
    
constraintMaster['speciesList'] = speciesList

#%% now go through each unique constraint and see if a constraint is shared by multiple nodes
# I defined a function to check if one node is the descendant of another

# first make a list of all the unique constraints, which are the combinations of C,T,L,S,Ca
constraintsAll = []
for i in range(len(constraintMaster)):
    constraintsAll.append(tuple(constraintMaster.iloc[i,1:6]))
constraintsUnique = list(set(constraintsAll))

# for all the unique constraints, get the index of the constraintsAll that has that constraint
indConstraint = np.zeros(len(constraintsAll))
for i in range(len(constraintsUnique)):
    # check that all 5 columns are the same
    ind = np.where(np.all(constraintMaster.iloc[:,1:6]==constraintsUnique[i],axis=1))[0]
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
for i in range(len(constraintsUnique)):
    tempNodeList = constraintMaster['node'].iloc[np.where(indConstraint==i)[0]].to_list()
    # get only the unique nodes, and we need the indices for the rest of the data
    _, ind = np.unique(tempNodeList,return_index=True)
    nodesUnique.append(np.array(constraintMaster['node'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    insideVarianceUnique.append(np.array(constraintMaster['insideVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    totalVarianceUnique.append(np.array(constraintMaster['totalVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    explainedVarianceUnique.append(np.array(constraintMaster['explainedVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    outsideVarianceUnique.append(np.array(constraintMaster['outsideVariance'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numMammalsUnique.append(np.array(constraintMaster['numMammals'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numBirdsUnique.append(np.array(constraintMaster['numBirds'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numReptilesUnique.append(np.array(constraintMaster['numReptiles'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])
    numAmphibiansUnique.append(np.array(constraintMaster['numAmphibians'].iloc[np.where(indConstraint==i)[0]].to_list())[ind])

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
    nodesUniqueKeep.append(nodesUniqueKeepTemp)
    insideVarianceKeep.append(insideVarianceKeepTemp)
    totalVarianceKeep.append(totalVarianceKeepTemp)
    explainedVarianceKeep.append(explainedVarianceKeepTemp)
    outsideVarianceKeep.append(outsideVarianceKeepTemp)
    numMammalsKeep.append(numMammalsKeepTemp)
    numBirdsKeep.append(numBirdsKeepTemp)
    numReptilesKeep.append(numReptilesKeepTemp)
    numAmphibiansKeep.append(numAmphibiansKeepTemp)
    
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
        
constraintMasterUnique = pd.DataFrame({'node':nodeList,'C':cervicalList,'T':thoracicList,'L':lumbarList,'S':sacralList,'Ca':caudalList,'insideVariance':insideVarianceList,'totalVariance':totalVarianceList,'explainedVariance':explainedVarianceList,'outsideVariance':outsideVarianceList,'numMammals':numMammalsList,'numBirds':numBirdsList,'numReptiles':numReptilesList,'numAmphibians':numAmphibiansList})

#%% cluster these unique constraints

methodName = 'complete'
# methodName = 'single'

corrMat = corrMatrixFunction(constraintMasterUnique,1,6)

# the pandas native way gives nans for some reason...
pdist = spc.distance.pdist(corrMat)
linkage = spc.linkage(pdist, method=methodName)
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
# add the cluster to the dataframe
constraintMasterUnique['cluster'] = idx
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
constraintMasterUnique_organizedTemp = constraintMasterUnique.iloc[newIndices,:]
constraintMasterUnique_organized = constraintMasterUnique_organizedTemp

corrMatConstraint_organized = corrMatrixFunction(constraintMasterUnique_organized,1,6)

# reset the index
constraintMasterUnique_organized.reset_index(inplace=True,drop=True)


#%% now do all of this again with the plasticity! (non-constraints)
# we already have the pca dataframe, so we just need to add the plasticity from the human-readable
# I think I should just include the most plastic from each node from the human-readable, otherwise there are some weird ones

# first the pca
# pcaPlasticity = pca[pca['explainedVariance'] >= plasticityThresholdLocal]
pcaPlasticity = pca[(pca['explainedVariance'] >= plasticityThresholdLocal) | (pca['insideVariance']/pca['outsideVariance'] > plasticityRatio)]
pcaPlasticity = pcaPlasticity.reset_index(drop=True)

#%% reorganize the dataframe so that we have a column for each node and a row for each species

methodName = 'complete'
# methodName = 'single'

corrMat = corrMatrixFunction(pcaPlasticity,1,6)

# the pandas native way gives nans for some reason...
pdist = spc.distance.pdist(corrMat)
linkage = spc.linkage(pdist, method=methodName)
idx = spc.fcluster(linkage, 0.25 * pdist.mean(), 'distance')
# add the cluster to the dataframe
pcaPlasticity['cluster'] = idx
# determine which cluster has the most constituents
unique, counts = np.unique(idx, return_counts=True)
# rearrange the tempMammal dataframe by these clusters starting from the most populous cluster
newIndices = []
countThresh = 3
indSort = np.argsort(counts)[::-1]
unique = unique[indSort]
counts = counts[indSort]
for i in range(len(unique)):
    if counts[i] >= countThresh:
        temp = np.where(idx==unique[i])[0]
        for j in range(len(temp)):
            newIndices.append(temp[j])
pcaPlasticity_organizedTemp = pcaPlasticity.iloc[newIndices,:]
pcaPlasticity_organized = pcaPlasticity_organizedTemp

corrMat_organized = corrMatrixFunction(pcaPlasticity_organized,1,6)

#%% make human-readable version of each of the big groups of plasticity

# first taking their average value
clusters = pcaPlasticity_organized['cluster'].unique()
meanPlasticity = np.zeros((len(clusters),len(dataCols)))
for i in range(len(clusters)):
    meanPlasticity[i,:] = np.mean(pcaPlasticity_organized[pcaPlasticity_organized['cluster']==clusters[i]].iloc[:,1:6].values,axis=0)
    
# make human-readable versions of each cluster plasticity:
cvec_unique = humanReadableConstraintMatrix(3,dataCols)

readablePlasticity = np.zeros((len(meanPlasticity),len(dataCols)))
for i in range(len(meanPlasticity)):
    readablePlasticity[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,meanPlasticity[i,:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
    
readablePlasticity = simplifyConstraints2(readablePlasticity)

# make an array with number of rows equal to the current dataframe
# and number of columns equal to dataCols
# and fill it with the human-readable constraints corresponding to the clusters in the dataframe
readablePlasticityArray = np.zeros((len(pcaPlasticity_organized),len(dataCols)))
for i in range(len(clusters)):
    readablePlasticityArray[pcaPlasticity_organized['cluster']==clusters[i],:] = readablePlasticity[i,:]

# add this to the dataframe
for i in range(len(dataCols)):
    pcaPlasticity_organized.insert(i+1,dataCols[i]+'_readable',readablePlasticityArray[:,i])

#%% # now do the human-readable : concatenate everything, keeping track of the nodes
# for now just do the nodes which are the same

if constraintName == 'vertebral':
    dataCols = ['C','T','L','S','Ca']
else:
    dataCols = toCheck

nodes = []
localVariance = []
outsideVariance = []
totalVariance = []
cervical = []
thoracic = []
lumbar = []
sacral = []
caudal = []
numMammals = []
numBirds = []
numReptiles = []
numAmphibians = []
for i in range(len(node1List)):
    if node1List[i] == node2List[i]:
        if node1List[i] == 'full':
            dirName = 'fullFormulaTree'
        else:
            dirName = 'subTree_'+node1List[i]+'_'+node2List[i]+'_speciesNum'+speciesNumList[i]
        # check if the file exists
        fileName = outputPath+constraintName+'/'+dirName+'/plasticity_humanReadable.csv'
        if not os.path.isfile(fileName):
            continue
        plasticity = pd.read_csv(fileName)
        # load the class data which is a list of strings
        classTemp = np.loadtxt(outputPath+constraintName+'/'+dirName+'/class.csv',delimiter=',',dtype=str)
        numMammalsTemp = np.sum(classTemp=='Mammalia')
        numBirdsTemp = np.sum(classTemp=='Aves')
        numReptilesTemp = np.sum(classTemp=='Reptilia')
        numAmphibiansTemp = np.sum(classTemp=='Amphibia')
        # for now just include the most plastic from each node
        indPlastic = np.argmax(plasticity['std'].values)
        plasticity = plasticity.iloc[indPlastic,:]
        nodes.append(node1List[i])
        localVariance.append(plasticity['std']**2)
        outsideVariance.append(plasticity['outside_std']**2)
        totalVariance.append(plasticity['totalVar'])
        numMammals.append(numMammalsTemp)
        numBirds.append(numBirdsTemp)
        numReptiles.append(numReptilesTemp)
        numAmphibians.append(numAmphibiansTemp)
        cervical.append(plasticity['Cervical'])
        thoracic.append(plasticity['Thoracic'])
        lumbar.append(plasticity['Lumbar'])
        sacral.append(plasticity['Sacral'])
        caudal.append(plasticity['Caudal'])

explainedVariancePercentage = np.array(np.array(localVariance)/np.array(totalVariance))
            
# make a dataframe of the loadings and explained variances
hrPlasticity = pd.DataFrame({'node':nodes,'C':cervical,'T':thoracic,'L':lumbar,'S':sacral,'Ca':caudal,'localVariance':localVariance,'outsideVariance':outsideVariance,'totalVariance':totalVariance,'explainedVariancePercentage':explainedVariancePercentage,'numMammals':numMammals,'numBirds':numBirds,'numReptiles':numReptiles,'numAmphibians':numAmphibians})

#%% find the actual inside and outside variance for each node
# the previous ones were normalized differently

outsideVariance = []
insideVariance = []
totalVariance = []
explainedVariance = []
for i in range(len(hrPlasticity)):
    node = hrPlasticity['node'].iloc[i]
    if node == 'full':
        outsideVariance.append(np.nan)
        insideVariance.append(np.var(np.dot(hrPlasticity.iloc[i,1:6].values,vertebralData.iloc[:,1:6].values.T)))
        totalVariance.append(np.sum(np.var(vertebralData.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
    else:
        # get the species in this node
        species = pd.read_csv(outputPath+constraintName+'/subTree_'+node+'_'+node+'_speciesNum'+str(hrPlasticity['numMammals'].iloc[i]+hrPlasticity['numBirds'].iloc[i]+hrPlasticity['numReptiles'].iloc[i]+hrPlasticity['numAmphibians'].iloc[i])+'/species.csv',header=None)[0].to_list()
        inside = vertebralData[vertebralData['species'].isin(species)]
        # dot product the pca loading by the data values and calculate the variance
        insideVariance.append(np.var(np.dot(hrPlasticity.iloc[i,1:6].values,inside.iloc[:,1:6].values.T))/np.sum(hrPlasticity.iloc[i,1:6].values**2))
        totalVariance.append(np.sum(np.var(inside.iloc[:,1:6].values.T,axis=1)))
        explainedVariance.append(insideVariance[-1]/totalVariance[-1])
        outside = vertebralData[~vertebralData['species'].isin(species)]
        outsideVariance.append(np.var(np.dot(hrPlasticity.iloc[i,1:6].values,outside.iloc[:,1:6].values.T))/np.sum(hrPlasticity.iloc[i,1:6].values**2))  
# rename the previous insideVariance and explainedVariance 
hrPlasticity.rename(columns={'localVariance':'localVarianceOld','outsideVariance':'outsideVarianceOld','totalVariance':'totalVarianceOld','explainedVariancePercentage':'explainedVariancePercentageOld'},inplace=True)
# add in the new ones after the 'Ca' column
# avoid the TypeError: loc must be int
indCol = int(np.where(pca.columns=='Ca')[0][0])
hrPlasticity.insert(indCol+1,'insideVariance',np.array(insideVariance))
hrPlasticity.insert(indCol+2,'totalVariance',np.array(totalVariance))
hrPlasticity.insert(indCol+3,'explainedVariance',np.array(explainedVariance))
hrPlasticity.insert(indCol+4,'outsideVariance',np.array(outsideVariance))


#%% filter the human-readable plasticity by the explained variance threshold

hrPlasticity = hrPlasticity[(hrPlasticity['explainedVariance'] >= plasticityThresholdLocal) | (hrPlasticity['insideVariance']/hrPlasticity['outsideVariance'] > plasticityRatio)]

#%% make a combined dataframe with the human-readable PCA (pcaPlasticity_organized) and the human-readable constraints (hrPlasticity)
# keep the following shared information:
# node, explainedVariancePercentage, numMammals, numBirds, numReptiles, numAmphibians, 
# C (_readable for pca), T (_readable for pca), L (_readable for pca), S (_readable for pca), Ca (_readable for pca)

plasticityMaster = pd.DataFrame()
# combine each of the shared columns from both dataframes
plasticityMaster['node'] = pd.concat([pcaPlasticity_organized['node'],hrPlasticity['node']],axis=0)
plasticityMaster['C'] = pd.concat([pcaPlasticity_organized['C_readable'],hrPlasticity['C']],axis=0)
plasticityMaster['T'] = pd.concat([pcaPlasticity_organized['T_readable'],hrPlasticity['T']],axis=0)
plasticityMaster['L'] = pd.concat([pcaPlasticity_organized['L_readable'],hrPlasticity['L']],axis=0)
plasticityMaster['S'] = pd.concat([pcaPlasticity_organized['S_readable'],hrPlasticity['S']],axis=0)
plasticityMaster['Ca'] = pd.concat([pcaPlasticity_organized['Ca_readable'],hrPlasticity['Ca']],axis=0)
plasticityMaster['insideVariance'] = pd.concat([pcaPlasticity_organized['insideVariance'],hrPlasticity['insideVariance']],axis=0)
plasticityMaster['totalVariance'] = pd.concat([pcaPlasticity_organized['totalVariance'],hrPlasticity['totalVariance']],axis=0)
plasticityMaster['explainedVariance'] = pd.concat([pcaPlasticity_organized['explainedVariance'],hrPlasticity['explainedVariance']],axis=0)
plasticityMaster['outsideVariance'] = pd.concat([pcaPlasticity_organized['outsideVariance'],hrPlasticity['outsideVariance']],axis=0)
plasticityMaster['numMammals'] = pd.concat([pcaPlasticity_organized['numMammals'],hrPlasticity['numMammals']],axis=0)
plasticityMaster['numBirds'] = pd.concat([pcaPlasticity_organized['numBirds'],hrPlasticity['numBirds']],axis=0)
plasticityMaster['numReptiles'] = pd.concat([pcaPlasticity_organized['numReptiles'],hrPlasticity['numReptiles']],axis=0)
plasticityMaster['numAmphibians'] = pd.concat([pcaPlasticity_organized['numAmphibians'],hrPlasticity['numAmphibians']],axis=0)

# reset the index
plasticityMaster.reset_index(inplace=True,drop=True)

# now go through each plasticity, and get the list of species corresponding to that plasticity's node
# and add that list to the plasticityMaster dataframe
speciesList = []
for i in range(len(plasticityMaster)):
    if plasticityMaster['node'].iloc[i] == 'full':
        speciesList.append(allSpecies)
    else:
        ind = node1List.index(plasticityMaster['node'].iloc[i])
        speciesList.append(np.loadtxt(outputPath+constraintName+'/subTree_'+node1List[ind]+'_'+node2List[ind]+'_speciesNum'+speciesNumList[ind]+'/species.csv',delimiter=',',dtype=str))
    
plasticityMaster['speciesList'] = speciesList

#%% now go through each unique plasticity and see if a plasticity is shared by multiple nodes
# I defined a function to check if one node is the descendant of another

# first make a list of all the unique plasticitys, which are the combinations of C,T,L,S,Ca
plasticityAll = []
for i in range(len(plasticityMaster)):
    plasticityAll.append(tuple(plasticityMaster.iloc[i,1:6]))
plasticityUnique = list(set(plasticityAll))

# for all the unique plasticity, get the index of the plasticityAll that has that plasticity
indPlasticity = np.zeros(len(plasticityAll))
for i in range(len(plasticityUnique)):
    # check that all 5 columns are the same
    ind = np.where(np.all(plasticityMaster.iloc[:,1:6]==plasticityUnique[i],axis=1))[0]
    indPlasticity[ind] = i
    
# get the list of nodes for each unique plasticity
nodesUnique = []
insideVarianceUnique = []
totalVarianceUnique = []
explainedVarianceUnique = []
outsideVarianceUnique = []
numMammalsUnique = []
numBirdsUnique = []
numReptilesUnique = []
numAmphibiansUnique = []
for i in range(len(plasticityUnique)):
    tempNodeList = plasticityMaster['node'].iloc[np.where(indPlasticity==i)[0]].to_list()
    # get only the unique nodes, and we need the indices for the rest of the data
    _, ind = np.unique(tempNodeList,return_index=True)
    nodesUnique.append(np.array(plasticityMaster['node'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    insideVarianceUnique.append(np.array(plasticityMaster['insideVariance'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    totalVarianceUnique.append(np.array(plasticityMaster['totalVariance'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    explainedVarianceUnique.append(np.array(plasticityMaster['explainedVariance'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    outsideVarianceUnique.append(np.array(plasticityMaster['outsideVariance'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    numMammalsUnique.append(np.array(plasticityMaster['numMammals'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    numBirdsUnique.append(np.array(plasticityMaster['numBirds'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    numReptilesUnique.append(np.array(plasticityMaster['numReptiles'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])
    numAmphibiansUnique.append(np.array(plasticityMaster['numAmphibians'].iloc[np.where(indPlasticity==i)[0]].to_list())[ind])

# now iterate through each unique plasticity, and then iterate through all the nodes
# keep only the nodes which have no ancestor node in the list of nodes for that plasticity
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
for i in range(len(plasticityUnique)):
    nodesUniqueKeepTemp = []
    insideVarianceKeepTemp = []
    totalVarianceKeepTemp = []
    explainedVarianceKeepTemp = []
    outsideVarianceKeepTemp = []
    numMammalsKeepTemp = []
    numBirdsKeepTemp = []
    numReptilesKeepTemp = []
    numAmphibiansKeepTemp = []
    # if 'full' is in the nodesUnique, then keep it and skip over using this function
    if 'full' in nodesUnique[i]:
        nodesUniqueKeepTemp.append('full')
        indFull = np.where(nodesUnique[i]=='full')[0][0]
        insideVarianceKeepTemp.append(insideVarianceUnique[i][indFull])
        totalVarianceKeepTemp.append(totalVarianceUnique[i][indFull])
        explainedVarianceKeepTemp.append(explainedVarianceUnique[i][indFull])
        outsideVarianceKeepTemp.append(outsideVarianceUnique[i][indFull])
        numMammalsKeepTemp.append(numMammalsUnique[i][indFull])
        numBirdsKeepTemp.append(numBirdsUnique[i][indFull])
        numReptilesKeepTemp.append(numReptilesUnique[i][indFull])
        numAmphibiansKeepTemp.append(numAmphibiansUnique[i][indFull])
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
    nodesUniqueKeep.append(nodesUniqueKeepTemp)
    insideVarianceKeep.append(insideVarianceKeepTemp)
    totalVarianceKeep.append(totalVarianceKeepTemp)
    explainedVarianceKeep.append(explainedVarianceKeepTemp)
    outsideVarianceKeep.append(outsideVarianceKeepTemp)
    numMammalsKeep.append(numMammalsKeepTemp)
    numBirdsKeep.append(numBirdsKeepTemp)
    numReptilesKeep.append(numReptilesKeepTemp)
    numAmphibiansKeep.append(numAmphibiansKeepTemp)
    
# now make a dataframe with the nodesUniqueKeep and the plasticityUnique
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
for i in range(len(plasticityUnique)):
    for j in range(len(nodesUniqueKeep[i])):
        nodeList.append(nodesUniqueKeep[i][j])
        cervicalList.append(plasticityUnique[i][0])
        thoracicList.append(plasticityUnique[i][1])
        lumbarList.append(plasticityUnique[i][2])
        sacralList.append(plasticityUnique[i][3])
        caudalList.append(plasticityUnique[i][4])
        insideVarianceList.append(insideVarianceKeep[i][j])
        totalVarianceList.append(totalVarianceKeep[i][j])
        explainedVarianceList.append(explainedVarianceKeep[i][j])
        outsideVarianceList.append(outsideVarianceKeep[i][j])
        numMammalsList.append(numMammalsKeep[i][j])
        numBirdsList.append(numBirdsKeep[i][j])
        numReptilesList.append(numReptilesKeep[i][j])
        numAmphibiansList.append(numAmphibiansKeep[i][j])
        
plasticityMasterUnique = pd.DataFrame({'node':nodeList,'C':cervicalList,'T':thoracicList,'L':lumbarList,'S':sacralList,'Ca':caudalList,'insideVariance':insideVarianceList,'totalVariance':totalVarianceList,'explainedVariance':explainedVarianceList,'outsideVariance':outsideVarianceList,'numMammals':numMammalsList,'numBirds':numBirdsList,'numReptiles':numReptilesList,'numAmphibians':numAmphibiansList})

#%% cluster these unique constraints

methodName = 'complete'
# methodName = 'single'

corrMat = corrMatrixFunction(plasticityMasterUnique,1,6)

# the pandas native way gives nans for some reason...
pdist = spc.distance.pdist(corrMat)
linkage = spc.linkage(pdist, method=methodName)
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
# add the cluster to the dataframe
plasticityMasterUnique['cluster'] = idx
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
plasticityMasterUnique_organizedTemp = plasticityMasterUnique.iloc[newIndices,:]
plasticityMasterUnique_organized = plasticityMasterUnique_organizedTemp

corrMatPlasticity_organized = corrMatrixFunction(plasticityMasterUnique_organized,1,6)

#%% save the dataframes

constraintMaster.to_csv(outputPath+'vertebral/constraintMaster.csv',index=False)
constraintMasterUnique_organized.to_csv(outputPath+'vertebral/constraintMasterUnique_organized.csv',index=False)
plasticityMaster.to_csv(outputPath+'vertebral/plasticityMaster.csv',index=False)
plasticityMasterUnique_organized.to_csv(outputPath+'vertebral/plasticityMasterUnique_organized.csv',index=False)