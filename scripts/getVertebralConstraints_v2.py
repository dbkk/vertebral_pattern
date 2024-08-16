# general script to get constraints given a tree, the list of species,
# and a list of their features of interest from a parameter file
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
import numpy as np
import os
import sympy as sp
import copy
import glob

#%% load our functions

import sys
sys.path.append(scriptPath)
from functionsPCA import *
from functionsIndividual import *
# from functionsMiscellany import *
from functionsHumanReadable import *

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
    
if os.path.exists(outputPath+constraintName+'/'):
    print('Base constraint directory exists. Continuing...')
else:
    print('Base constraint directory does not exist. Creating...')
    os.mkdir(outputPath+constraintName+'/')
    
#%% load the tree

# load the only .nwk file in the inputPath, so search for it with glob
treePath = glob.glob(inputPath+'*.nwk')[0]

with open(treePath) as f:
    treeFile = f.read()
treeInitial = TreeNode.read(StringIO(treeFile))
    
#%% load the species list and the features

# load the only .csv file in the inputPath, so search for it with glob
dataPath = glob.glob(inputPath+'vertebralFormulaOrdered_v2.csv')[0]
featureData = pd.read_csv(dataPath)

speciesFeatures = list(set(featureData['Species'].to_list()))

#%% reduce the tree to only the species in the feature list and remove species not in the tree

# species in tree
speciesInTree = []
numTips = 0
for tips in treeInitial.tips():
    numTips = numTips + 1
    speciesInTree.append(tips.name)
    
namesNotInFeatures = list(set(speciesFeatures) - set(speciesInTree))
species = list(set(speciesInTree) & set(speciesFeatures))

# reduce tree
tree = treeInitial.shear(species)
tree.prune()

# reduce features
features = featureData[featureData['Species'].isin(species)]
features.reset_index(inplace=True,drop=True)

#%% get the unique list of the subgroup if defined

if np.size(subgroup) > 0:
    if subgroup in featureData.columns:
        groupList = list(set(features[subgroup].unique()))
    else:
        groupList = 0
else:
    groupList = 0
    
# I actually want my own ordering here for my color scheme:
groupList = (['Mammalia','Aves','Reptilia','Amphibia'])

#%% get lists of nodes and their descendant tip names and numbers

nodeName = []
nodeChildrenTips = []
nodeChildrenTipNumber = []
for node in tree.non_tips():
    nodeName.append(node.name)
    cnt = 0
    temp = []
    for tips in node.tips():
        temp.append(tips.name)
        cnt = cnt + 1
    nodeChildrenTipNumber.append(cnt)
    nodeChildrenTips.append(temp)

nodeChildrenTipNumber = np.asarray(nodeChildrenTipNumber, dtype="object")
nodeChildrenTips = np.asarray(nodeChildrenTips, dtype="object")
nodeName = np.asarray(nodeName, dtype="object")

#%% remove all that are less than the "numThresh" and sort

nodeNameThresh = nodeName[nodeChildrenTipNumber>=numThresh]
nodeChildrenTipsThresh = nodeChildrenTips[nodeChildrenTipNumber>=numThresh]
nodeChildrenTipNumberThresh = nodeChildrenTipNumber[nodeChildrenTipNumber>=numThresh]

# sort
indSort = np.flip(np.argsort(nodeChildrenTipNumberThresh))
nodeChildrenTipNumberThresh = nodeChildrenTipNumberThresh[indSort]
nodeChildrenTipsThresh = nodeChildrenTipsThresh[indSort]
nodeNameThresh = nodeNameThresh[indSort]

#%% take care of nans if any according to the fillNaNs parameter
        
if fillNaNs: # replace nans with mean value of that column
    features[pd.Index(toCheck)] = features[pd.Index(toCheck)].fillna(features[pd.Index(toCheck)].mean())
    # there may still be columns that are completely nan, so set those to zero
    features[pd.Index(toCheck)] = features[pd.Index(toCheck)].fillna(0)
    # may need to do it again
    features[pd.Index(toCheck)] = features[pd.Index(toCheck)].fillna(features[pd.Index(toCheck)].mean())
else: # drop those rows where there are any nans in the toCheck columns
    features = features.dropna(subset=pd.Index(toCheck))
    features.reset_index(inplace=True,drop=True)
    
#%% full tree and getting global values

# going to use the global (full tree) values of the standard deviation and mean to normalize all subtrees
globalStd = np.nanstd(features.loc[:,pd.Index(toCheck)],axis=0)
globalMean = np.nanmean(features.loc[:,pd.Index(toCheck)],axis=0)
globalX = features.loc[:,pd.Index(toCheck)].values # need this for later
globalXsc = (globalX-globalMean)/globalStd # need this for later
globalSpecies = features['Species'].to_list()

# make the save path

savePathBase = outputPath+constraintName+'/'
savePath = savePathBase+'fullFormulaTree/'
if not os.path.exists(savePath):
    os.mkdir(savePath)

# run the PCA
X,Xpca,loadings,explained,explainedPercentage,xSubgroup = constraintPCA(features,tree,species,constraintName,toCheck,subgroup,groupList,globalMean,globalStd,maxPCA,savePath)
# get the rank according to our criteria
rank,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal = getRankAndConstraints(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck,maxPCA)
# plot the PC1 vs PC2
plotPC1PC2(Xpca,groupList,xSubgroup,subgroup,savePath,explainedPercentage,rank,constraintName)
# plot the loadings
plotLoading(loadings,explainedPercentage,savePath,rank,constraintName,toCheck)
# get the relative variance of individual features
_,var,_,constraintsIndividual,nonConstraintsIndividual,constraintsIndividualLocal,nonConstraintsIndividualLocal = constraintIndividual(features,tree,species,globalSpecies,constraintName,toCheck,subgroup,groupList,savePath,constraintRatio,plasticityRatio,constraintRatioHumanReadable,constraintThresholdLocal)
# plot the relative variance of individual features
plotVariance(var,savePath,constraintName,toCheck)

# plot local PCA constraints
if np.sum(constraintsLocal) > 0: # may be some constraints!
    analyzeConstraintsLocal(X,Xpca,globalSpecies,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,constraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)
if np.sum(nonConstraintsLocal) > 0: # may be some plastic features!
    analyzeNonConstraintsLocal(X,Xpca,globalSpecies,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,nonConstraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)

# human readable constraints
if len(toCheck) < 20: # otherwise the dimension is too high...
    cvec_unique = humanReadableConstraintMatrix(spanHumanReadable,toCheck)
    
    # analyze the PCA results and make them "human-readable"
    candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,cvec_unique,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck)
    # analyze and plot these "human-readable" PCA constraints if there are any
    if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
        analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,species,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,features['Class'].to_list(),features['Class'].to_list(),savePath,toCheck)
    
    candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std = constraintHumanReadable(X.values,globalSpecies,X,globalSpecies,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,plasticityThresholdLocal,toCheck,cvec_unique,savePath,spanHumanReadable)
    # plot the human readable constraints
    if np.size(candidates)>0:
        plotReadableConstraints(candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,X.values,globalSpecies,X,globalSpecies,toCheck,features['Class'].to_list(),features['Class'].to_list(),savePath,toCheck)
    if np.size(candidates_plasticity)>0:
        plotReadablePlasticity(candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std,X.values,globalSpecies,X,globalSpecies,toCheck,features['Class'].to_list(),savePath,toCheck)

else:
    # analyze the PCA results and make them "human-readable" by rounding
    candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable2(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck)
    # analyze and plot these "human-readable" PCA constraints if there are any
    if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
        analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,species,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,features['Class'].to_list(),features['Class'].to_list(),savePath,toCheck)
    
# we can check the individual ones too...
if np.sum(np.sum(constraintsIndividualLocal)) > 0: # may be some individual constraints!
    analyzeConstraintsIndividualLocal(features,var,globalSpecies,globalSpecies,groupList,xSubgroup,subgroup,savePath,constraintsIndividualLocal,toCheck,constraintName)
if np.sum(np.sum(nonConstraintsIndividualLocal)) > 0: # may be some individual constraints!
    analyzeNonConstraintsIndividualLocal(features,var,globalSpecies,globalSpecies,groupList,xSubgroup,subgroup,savePath,nonConstraintsIndividualLocal,toCheck,constraintName)

# save features list
np.savetxt(savePath+'features.csv',toCheck,fmt='%s') # save feature list

#%% main loop

# now loop through all the nodes with more than numThresh tips, comparing also subtrees with another subtree of that tree removed

allRanks = []
allRanks.append(rank)
numSpecies = []
numSpecies.append(len(Xpca))

subtree1 = []
subtree1.append(-1)
subtree2 = []
subtree2.append(-1)

allConstraints = []
allConstraints.append(np.sum(constraints))
allNonConstraints = []
allNonConstraints.append(np.sum(constraints))

if len(toCheck) < 20: # otherwise the dimension is too high...
    allHRConstraints = []
    allHRConstraints.append(len(candidates))
    allHRNonConstraints = []
    allHRNonConstraints.append(len(candidates_plasticity))

allConstraintsIndividual = []
allConstraintsCumsum = []
allConstraintsCumsumReverse = []
allNonConstraintsIndividual = []
allNonConstraintsCumsum = []
allNonConstraintsCumsumReverse = []
allConstraintsIndividual.append(np.sum(constraintsIndividual[0,:]))
allConstraintsCumsum.append(np.sum(constraintsIndividual[1,:]))
allConstraintsCumsumReverse.append(np.sum(constraintsIndividual[2,:]))
allNonConstraintsIndividual.append(np.sum(nonConstraintsIndividual[0,:]))
allNonConstraintsCumsum.append(np.sum(nonConstraintsIndividual[1,:]))
allNonConstraintsCumsumReverse.append(np.sum(nonConstraintsIndividual[2,:]))

for k in range(0,len(nodeChildrenTipsThresh)):
    for p in range(k,k+1):#len(nodeChildrenTipsThresh)):
        
        print(f"{k}th subtree vs. {p}th subtree: {nodeNameThresh[k]} vs. {nodeNameThresh[p]}")
        
        if p == k: # comparing with itself
            
            # get subtree k's species and save path
            subtreeSpeciesList = nodeChildrenTipsThresh[k]
            subtreeSavePath = savePathBase+'subTree_'+nodeNameThresh[k]+'_'+nodeNameThresh[k]+'_speciesNum'+str(len(subtreeSpeciesList))+'/'
            if not os.path.exists(subtreeSavePath):
                os.mkdir(subtreeSavePath)
                
            # find those in the subtreeSpeciesList which are not in the features['Species'] list
            difference = list(set(subtreeSpeciesList)-set(species))
                
            # run the PCA
            X,Xpca,loadings,explained,explainedPercentage,xSubgroup = constraintPCA(features,tree,subtreeSpeciesList,constraintName,toCheck,subgroup,groupList,globalMean,globalStd,maxPCA,subtreeSavePath)            
            # get the rank according to our criteria
            rank,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal = getRankAndConstraints(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck,maxPCA)
            # plot the PC1 vs PC2
            plotPC1PC2(Xpca,groupList,xSubgroup,subgroup,subtreeSavePath,explainedPercentage,rank,constraintName)
            # plot the loadings
            plotLoading(loadings,explainedPercentage,subtreeSavePath,rank,constraintName,toCheck)
            # get the relative variance of individual features
            _,var,_,constraintsIndividual,nonConstraintsIndividual,constraintsIndividualLocal,nonConstraintsIndividualLocal = constraintIndividual(features,tree,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,subtreeSavePath,constraintRatio,plasticityRatio,constraintRatioHumanReadable,constraintThresholdLocal)
            # plot the relative variance of individual features
            plotVariance(var,subtreeSavePath,constraintName,toCheck)
            # human readable constraints
            if len(toCheck) < 20: # otherwise the dimension is too high...
                
                # analyze the PCA results and make them "human-readable"
                candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,cvec_unique,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck)
                # analyze and plot these "human-readable" PCA constraints if there are any
                if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
                    analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                
                candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std = constraintHumanReadable(X.values,subtreeSpeciesList,features,globalSpecies,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,plasticityThresholdLocal,toCheck,cvec_unique,subtreeSavePath,spanHumanReadable)
                # plot the human readable constraints
                if np.size(candidates)>0:
                    plotReadableConstraints(candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,X.values,subtreeSpeciesList,features,globalSpecies,toCheck,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                if np.size(candidates)>0:
                    plotReadablePlasticity(candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std,X.values,subtreeSpeciesList,features,globalSpecies,toCheck,xSubgroup,subtreeSavePath,toCheck)

            else:
                # analyze the PCA results and make them "human-readable" by rounding
                candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable2(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck)
                # analyze and plot these "human-readable" PCA constraints if there are any
                if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
                    analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                

            # save features list
            np.savetxt(subtreeSavePath+'features.csv',toCheck,fmt='%s') # save feature list
            
        else:
        
            # determine which one is the smaller subtree and whether it is a subset or not (if not len(difference) is 0!)
            if len(nodeChildrenTipsThresh[k]) > len(nodeChildrenTipsThresh[p]):
                difference = list(set(nodeChildrenTipsThresh[k])-set(nodeChildrenTipsThresh[p]))
                if (len(difference) == len(nodeChildrenTipsThresh[k])) | (len(difference) < numThresh):
                    continue
            else: 
                difference = list(set(nodeChildrenTipsThresh[p])-set(nodeChildrenTipsThresh[k]))
                if (len(difference) == len(nodeChildrenTipsThresh[p])) | (len(difference) < numThresh):
                    continue
                
            # get subtree k's species and save path
            subtreeSpeciesList = difference
            subtreeSavePath = savePathBase+'subTree_'+nodeNameThresh[k]+'_'+nodeNameThresh[p]+'_speciesNum'+str(len(subtreeSpeciesList))+'/'
            if not os.path.exists(subtreeSavePath):
                os.mkdir(subtreeSavePath)
                
            # run the PCA
            X,Xpca,loadings,explained,explainedPercentage,xSubgroup = constraintPCA(features,tree,subtreeSpeciesList,constraintName,toCheck,subgroup,groupList,globalMean,globalStd,maxPCA,subtreeSavePath)            
            # get the rank according to our criteria
            rank,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal = getRankAndConstraints(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck,maxPCA)
            # plot the PC1 vs PC2
            plotPC1PC2(Xpca,groupList,xSubgroup,subgroup,subtreeSavePath,explainedPercentage,rank,constraintName)
            # plot the loadings
            plotLoading(loadings,explainedPercentage,subtreeSavePath,rank,constraintName,toCheck)
            # get the relative variance of individual features
            _,var,_,constraintsIndividual,nonConstraintsIndividual,constraintsIndividualLocal,nonConstraintsIndividualLocal = constraintIndividual(features,tree,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,subtreeSavePath,constraintRatio,plasticityRatio,constraintRatioHumanReadable,constraintThresholdLocal)
            # plot the relative variance of individual features
            plotVariance(var,subtreeSavePath,constraintName,toCheck)
            # human readable constraints
            if len(toCheck) < 20: # otherwise the dimension is too high...
                
                # analyze the PCA results and make them "human-readable"
                candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,cvec_unique,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal)
                # analyze and plot these "human-readable" PCA constraints if there are any
                if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
                    analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                
                candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std = constraintHumanReadable(X.values,subtreeSpeciesList,features,globalSpecies,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,plasticityThresholdLocal,toCheck,cvec_unique,subtreeSavePath,spanHumanReadable)
                # plot the human readable constraints
                if np.size(candidates)>0:
                    plotReadableConstraints(candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,X.values,subtreeSpeciesList,features,globalSpecies,toCheck,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                if np.size(candidates)>0:
                    plotReadablePlasticity(candidates_plasticity,candidates_plasticity_std,candidates_plasticity_outside_std,X.values,subtreeSpeciesList,features,globalSpecies,toCheck,xSubgroup,subtreeSavePath,toCheck)

            else:
                # analyze the PCA results and make them "human-readable" by rounding
                candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal = makePCAReadable2(Xpca,subtreeSpeciesList,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,subtreeSavePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck)
                # analyze and plot these "human-readable" PCA constraints if there are any
                if (len(candidateConstraints) > 0) | (len(candidateConstraintsLocal) > 0):
                    analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,X.values,features,subtreeSpeciesList,globalSpecies,constraintName,toCheck,subgroup,groupList,constraintRatioHumanReadable,plasticityRatio,constraintThresholdLocal,xSubgroup,features['Class'].to_list(),subtreeSavePath,toCheck)
                

            # save features list
            np.savetxt(subtreeSavePath+'features.csv',toCheck,fmt='%s') # save feature list
        
        subtree1.append(nodeNameThresh[k])
        subtree2.append(nodeNameThresh[p])
        numSpecies.append(len(Xpca))
        allRanks.append(rank)
        allConstraints.append(np.sum(constraints))
        allNonConstraints.append(np.sum(nonConstraints))
        allConstraintsIndividual.append(np.sum(constraintsIndividual[0,:]))
        allConstraintsCumsum.append(np.sum(constraintsIndividual[1,:]))
        allConstraintsCumsumReverse.append(np.sum(constraintsIndividual[2,:]))
        allNonConstraintsIndividual.append(np.sum(nonConstraintsIndividual[0,:]))
        allNonConstraintsCumsum.append(np.sum(nonConstraintsIndividual[1,:]))
        allNonConstraintsCumsumReverse.append(np.sum(nonConstraintsIndividual[2,:]))
        if len(toCheck) < 20: # otherwise the dimension is too high...
            allHRConstraints.append(len(candidates))
            allHRNonConstraints.append(len(candidates_plasticity))

        # do a little more work if there seem to be constraints!
        
        if np.sum(constraints) > 0: # may be some constraints!
            
            analyzeConstraints(X,Xpca,subtreeSpeciesList,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,explained,explainedPercentage,rank,constraints,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)
            
        if np.sum(nonConstraints) > 0: # may be some plastic features!
            
            analyzeNonConstraints(X,Xpca,subtreeSpeciesList,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,explained,explainedPercentage,rank,nonConstraints,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)
            
        if np.sum(constraintsLocal) > 0: # may be some constraints!
            
            analyzeConstraintsLocal(X,Xpca,subtreeSpeciesList,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,explained,explainedPercentage,rank,constraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)
            
        if np.sum(nonConstraintsLocal) > 0: # may be some plastic features!
            
            analyzeNonConstraintsLocal(X,Xpca,subtreeSpeciesList,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,explained,explainedPercentage,rank,nonConstraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings)
            
        # analyze individual constraints and non-constraints

        if np.sum(np.sum(constraintsIndividual)) > 0: # may be some individual constraints!
        
            analyzeConstraintsIndividual(features,var,subtreeSpeciesList,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,constraintsIndividual,toCheck,constraintName)
            
        if np.sum(np.sum(nonConstraintsIndividual)) > 0: # may be some individual constraints!
            
            analyzeNonConstraintsIndividual(features,var,subtreeSpeciesList,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,nonConstraintsIndividual,toCheck,constraintName)
            
        if np.sum(np.sum(constraintsIndividualLocal)) > 0: # may be some individual constraints!
        
            analyzeConstraintsIndividualLocal(features,var,subtreeSpeciesList,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,constraintsIndividualLocal,toCheck,constraintName)
            
        if np.sum(np.sum(nonConstraintsIndividualLocal)) > 0: # may be some individual constraints!
            
            analyzeNonConstraintsIndividualLocal(features,var,subtreeSpeciesList,globalSpecies,groupList,xSubgroup,subgroup,subtreeSavePath,nonConstraintsIndividualLocal,toCheck,constraintName)
            
                    
dfRank = pd.DataFrame()
dfRank['subtree1'] = subtree1
dfRank['subtree2'] = subtree2
dfRank['numSpecies'] = np.array(numSpecies).astype(int)
dfRank['numConstraints'] = np.array(allConstraints).astype(int)
dfRank['numPlasticity'] = np.array(allNonConstraints).astype(int)
dfRank['rank'] = np.array(allRanks).astype(int)
if len(toCheck) < 20: # otherwise the dimension is too high...
    dfRank['numConstraintsHumanReadable'] = np.array(allHRConstraints).astype(int)
    dfRank['numPlasticityHumanReadable'] = np.array(allHRNonConstraints).astype(int)
dfRank['numConstraintsIndivual'] = np.array(allConstraintsIndividual).astype(int)
dfRank['numConstraintsCumsum'] = np.array(allConstraintsCumsum).astype(int)
dfRank['numConstraintsCumsumReverse'] = np.array(allConstraintsCumsumReverse).astype(int)
dfRank['numPlasticityIndivual'] = np.array(allNonConstraintsIndividual).astype(int)
dfRank['numPlasticityCumsum'] = np.array(allNonConstraintsCumsum).astype(int)
dfRank['numPlasticityCumsumReverse'] = np.array(allNonConstraintsCumsumReverse).astype(int)
dfRank.to_csv(savePathBase+'subtreeRanks.csv',index=False)