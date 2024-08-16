# PCA functions for the vertebral paper

#%% libraries

from sklearn.decomposition import PCA
from skbio import TreeNode
from io import StringIO
import pandas as pd
import numpy as np
import sympy as sp

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

#%% function to simplify constraints used with the analysis

def simplifyConstraints(x):

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

#%% function that runs the PC analysis given a list of species and their features

def constraintPCA(df,tree,species,constraintName,toCheck,subgroup,groupList,globalMean,globalStd,maxPCA,savePath):
    
    # get the specific values we want
    X = df.loc[df['Species'].isin(species),toCheck]
    if groupList == 0:
        xSubgroup = 0
    else:
        xSubgroup = df.loc[df['Species'].isin(species),subgroup].squeeze().tolist()
    xSpecies = df.loc[df['Species'].isin(species),'Species'].squeeze().tolist()
        
    # normalize
    Xsc = (X.values-globalMean)/globalStd
    Xsc = np.nan_to_num(Xsc)
    
    # pca
    pca = PCA(n_components = np.min([len(toCheck),maxPCA]))
    Xpca = pca.fit_transform(Xsc) # fit_transform will project the data possibly differently than just multiplying the loading matrix by the data
    loadings = pca.components_
    explained = pca.explained_variance_
    explainedPercentage = pca.explained_variance_ratio_
    
    # save the loadings, etc.
    np.savetxt(savePath+'pca.csv',Xpca,delimiter=',') # save pca data
    np.savetxt(savePath+'loadings.csv',loadings,delimiter=',') # save loadings
    np.savetxt(savePath+'explainedVariance.csv',explained,delimiter=',') # the explained variance
    np.savetxt(savePath+'explainedPercentage.csv',explainedPercentage,delimiter=',')
    if groupList != 0:
        np.savetxt(savePath+subgroup.lower()+'.csv',xSubgroup,fmt='%s')
    np.savetxt(savePath+'species.csv',xSpecies,fmt='%s')
    data = pd.DataFrame()
    data['species'] = xSpecies
    # add X to the data frame
    for i in range(len(toCheck)):
        data[toCheck[i]] = X[toCheck[i]].values
    # np.savetxt(savePath+constraintName.lower()+'Data.csv',X,delimiter=',')
    data.to_csv(savePath+constraintName.lower()+'Data.csv',index=False)
    
    # get the subtree 
    subtree = tree.shear(list(xSpecies))
    subtree.prune()
    subtree.write(savePath+'tree.nwk',format='newick')
    
    return(X,Xpca,loadings,explained,explainedPercentage,xSubgroup)


#%% function to determine the rank and constraints according to our thresholds

def getRankAndConstraints(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck,maxPCA):
    
    rank = np.min([len(toCheck),maxPCA]) # initialize
    constraints = np.zeros(np.min([len(toCheck),maxPCA]))
    nonConstraints = np.zeros(np.min([len(toCheck),maxPCA]))
    globalCov = np.zeros(np.min([len(toCheck),maxPCA]))
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    
    # checking globally
    if len(species) < len(globalSpecies):
        for i in range(np.min([len(toCheck),maxPCA])):
            if len(species) < len(globalSpecies):
                temp = np.nansum(loadings[i,:]*globalXsc,axis=1)
                tempFiltered = np.delete(temp, indRemove, axis=0)
                tempCov = np.cov(tempFiltered)
                globalCov[i] = tempCov
                if tempCov > constraintRatio*explained[i]:
                    rank = rank - 1
                    constraints[i] = 1
                if explained[i] > plasticityRatio*tempCov:
                    nonConstraints[i] = 1
            else:
                if explainedPercentage[i] < 1/constraintRatio:
                    rank = rank - 1
                    constraints[i] = 1
                if explainedPercentage[i] > 1-(1/constraintRatio): # this basically can't happen...
                    nonConstraints[i] = 1
                
    constraintsLocal = np.zeros(np.min([len(toCheck),maxPCA]))
    nonConstraintsLocal = np.zeros(np.min([len(toCheck),maxPCA]))
                
    # checking locally
    for i in range(np.min([len(toCheck),maxPCA])):
        if explainedPercentage[i] < constraintThresholdLocal:
            constraintsLocal[i] = 1
        if explainedPercentage[i] > 1-constraintThresholdLocal:
            nonConstraintsLocal[i] = 1
            
    return(rank,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal)
            

#%% make the PCA result human-readable

def makePCAReadable(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,cvec_unique,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck):

    # first the global constraints
    indConstraints = np.where(constraints==1)[0]
    candidateConstraints = np.zeros((len(indConstraints),len(toCheck)))
    for i in range(len(indConstraints)):
        candidateConstraints[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,loadings[indConstraints[i],:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
        
    candidateConstraints = simplifyConstraints(candidateConstraints)
    candidateConstraints += 0.
    
    # next global plasticity
    indNonConstraints = np.where(nonConstraints==1)[0]
    candidateNonConstraints = np.zeros((len(indNonConstraints),len(toCheck)))
    for i in range(len(indNonConstraints)):
        candidateNonConstraints[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,loadings[indNonConstraints[i],:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
    
    candidateNonConstraints = simplifyConstraints(candidateNonConstraints)
    candidateNonConstraints += 0.
        
    # next local constraints
    indConstraintsLocal = np.where(constraintsLocal==1)[0]
    candidateConstraintsLocal = np.zeros((len(indConstraintsLocal),len(toCheck)))
    for i in range(len(indConstraintsLocal)):
        candidateConstraintsLocal[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,loadings[indConstraintsLocal[i],:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
        
    candidateConstraintsLocal = simplifyConstraints(candidateConstraintsLocal)
    candidateConstraintsLocal += 0.
    
    # next local plasticity
    indNonConstraintsLocal = np.where(nonConstraintsLocal==1)[0]
    candidateNonConstraintsLocal = np.zeros((len(indNonConstraintsLocal),len(toCheck)))
    for i in range(len(indNonConstraintsLocal)):
        candidateNonConstraintsLocal[i,:] = cvec_unique[np.argmax(np.abs(np.dot(cvec_unique,loadings[indNonConstraintsLocal[i],:])/(np.linalg.norm(cvec_unique,axis=1)))),:]
        
    candidateNonConstraintsLocal = simplifyConstraints(candidateNonConstraintsLocal)
    candidateNonConstraintsLocal += 0.
    
    return(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal)
    
#%% another function to the make the pca readable in case the dimensions of cvec_unique are impossibly large
# in this case we'll just round the loadings!
    
def makePCAReadable2(Xpca,species,globalXsc,globalSpecies,loadings,explained,explainedPercentage,constraints,nonConstraints,globalCov,constraintsLocal,nonConstraintsLocal,savePath,constraintRatio,plasticityRatio,constraintThresholdLocal,toCheck):

    valuesAllowed = 1

    # first the global constraints
    indConstraints = np.where(constraints==1)[0]
    candidateConstraints = np.zeros((len(indConstraints),len(toCheck)))
    for i in range(len(indConstraints)):
        # first divide by the largest element
        candidateConstraints[i,:] = loadings[indConstraints[i],:]/np.max(np.abs(loadings[indConstraints[i],:]))
        # now we only allow up to values of 1
        candidateConstraints[i,:] = np.round(candidateConstraints[i,:]*valuesAllowed)/valuesAllowed
        
    candidateConstraints = simplifyConstraints(candidateConstraints)+.0
    
    # next global plasticity
    indNonConstraints = np.where(nonConstraints==1)[0]
    candidateNonConstraints = np.zeros((len(indNonConstraints),len(toCheck)))
    for i in range(len(indNonConstraints)):
        # first divide by the largest element
        candidateNonConstraints[i,:] = loadings[indNonConstraints[i],:]/np.max(np.abs(loadings[indNonConstraints[i],:]))
        # now we only allow up to values of 2
        candidateNonConstraints[i,:] = np.round(candidateNonConstraints[i,:]*valuesAllowed)/valuesAllowed
    
    candidateNonConstraints = simplifyConstraints(candidateNonConstraints)+.0
    
    # next local constraints
    indConstraintsLocal = np.where(constraintsLocal==1)[0]
    candidateConstraintsLocal = np.zeros((len(indConstraintsLocal),len(toCheck)))
    for i in range(len(indConstraintsLocal)):
        # first divide by the largest element
        candidateConstraintsLocal[i,:] = loadings[indConstraintsLocal[i],:]/np.max(np.abs(loadings[indConstraintsLocal[i],:]))
        # now we only allow up to values of 2
        candidateConstraintsLocal[i,:] = np.round(candidateConstraintsLocal[i,:]*valuesAllowed)/valuesAllowed
        
    candidateConstraintsLocal = simplifyConstraints(candidateConstraintsLocal)+.0
    
    # next local plasticity
    indNonConstraintsLocal = np.where(nonConstraintsLocal==1)[0]
    candidateNonConstraintsLocal = np.zeros((len(indNonConstraintsLocal),len(toCheck)))
    for i in range(len(indNonConstraintsLocal)):
        # first divide by the largest element
        candidateNonConstraintsLocal[i,:] = loadings[indNonConstraintsLocal[i],:]/np.max(np.abs(loadings[indNonConstraintsLocal[i],:]))
        # now we only allow up to values of 2
        candidateNonConstraintsLocal[i,:] = np.round(candidateNonConstraintsLocal[i,:]*valuesAllowed)/valuesAllowed
        
    candidateNonConstraintsLocal = simplifyConstraints(candidateNonConstraintsLocal)+.0

    
    return(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal)
    
#%% analyze and plot the human-readable PCA results

def analyzePlotPCAreadable(candidateConstraints,candidateNonConstraints,candidateConstraintsLocal,candidateNonConstraintsLocal,loadings,explained,globalMean,globalStd,v_counts_subtree,full_tree,species,globalSpecies,constraintName,featureNames,subgroup,groupList,constraintRatioHumanReadable,plasticityRatioHumanReadable,constraintThresholdLocal,class_subtree,class_global,savePath,toCheck):
    
    # we want to also do something for the full tree (so when len(globalSpecies) = 0)
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    v_counts_outside = full_tree.loc[:,pd.Index(toCheck)].values
    if len(indRemove)<len(globalSpecies):
        v_counts_outside = np.delete(v_counts_outside, indRemove, axis=0)
        class_global = np.delete(class_global, indRemove, axis=0)
        
    Xsc = (v_counts_subtree-globalMean)/globalStd
    Xsc = np.nan_to_num(Xsc)

    # make a nice constraint name
    if featureNames[0]=='Cervical':
        constraintNames = ['C','T','L','S','Ca']
    else:
        constraintNames = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                constraintNames.append(hoxLetters[i]+str(j+1))
        
    # get the number in each unique class for the subtree
    uniqueClasses = np.unique(class_subtree)
    numClasses = np.zeros(len(uniqueClasses))
    for i in range(len(uniqueClasses)):
        numClasses[i] = np.sum(np.array(class_subtree)==uniqueClasses[i])
    textClass = ''
    for j in range(len(uniqueClasses)):
        textClass = textClass + uniqueClasses[j] + ' (' + str(int(numClasses[j])) + ')' + ', '
    textClass = textClass[:-2]
    
    # get the number in each unique class for the full tree
    uniqueClassesGlobal = np.unique(class_global)
    numClassesGlobal = np.zeros(len(uniqueClassesGlobal))
    for i in range(len(uniqueClassesGlobal)):
        numClassesGlobal[i] = np.sum(np.array(class_global)==uniqueClassesGlobal[i])
    textClassGlobal = ''
    for j in range(len(uniqueClassesGlobal)):
        textClassGlobal = textClassGlobal + uniqueClassesGlobal[j] + ' (' + str(int(numClassesGlobal[j])) + ')' + ', '
    textClassGlobal = textClassGlobal[:-2]
    
    markerWheelLocal = ['o','s','^','>'] # one for each class
        
    classWheel = groupList
            
    # first the "global" ones
    candidateConstraints_std = []
    candidateConstraints_outside_std = []
    for i in range(len(candidateConstraints)):
                    
        constraintText = ''
        for j in range(len(candidateConstraints[i])):
            if candidateConstraints[i][j]!=0:
                if candidateConstraints[i][j]>0:
                    constraintText = constraintText + ' + '
                else:
                    constraintText = constraintText + ' - '
                constraintText = constraintText + str(np.abs(int(candidateConstraints[i][j]))) + constraintNames[j]
        constraintText = constraintText[3:]

        ind_outside = np.linspace(0,len(v_counts_outside)-1,len(v_counts_subtree)).astype(int)
        values_subtree = np.zeros(len(v_counts_subtree))
        values_outside = np.zeros(len(v_counts_subtree))
        values_global = np.zeros(len(v_counts_outside))
        indNonZero = np.where(candidateConstraints[i,:]!=0)[0]
        values_subtree = np.sum(candidateConstraints[i,:]*v_counts_subtree,axis=1)
        values_outside = np.sum(candidateConstraints[i,:]*v_counts_outside[ind_outside,:],axis=1)
        values_global = np.sum(candidateConstraints[i,:]*v_counts_outside,axis=1)
            
        # need to normalize by the "std" of the constraint formula itself like in PCA
        constraintStdNorm = np.sqrt(np.sum(np.square(candidateConstraints[i,:])))
            
        candidates_std = np.nanstd(values_subtree)/constraintStdNorm
        candidates_outside_std = np.nanstd(values_global)/constraintStdNorm
        candidateConstraints_std.append(candidates_std)
        candidateConstraints_outside_std.append(candidates_outside_std)

        fig, ax = plt.subplots(figsize=(5,5))
            
        for j in range(len(uniqueClasses)):
            indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
            plt.plot(values_outside[indClass],values_subtree[indClass],color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],marker=markerWheelLocal[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],linestyle='None',alpha=0.5,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
        plt.ylim(np.min([0,np.min(values_outside)-1]),np.max([np.max(values_subtree)+1,np.max(values_outside)+1]))
        plt.xlabel(f'values outside subtree for constraint: {constraintText}')
        plt.ylabel(f'values inside subtree for constraint: {constraintText}')
        
        plt.legend(frameon=False)

        plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma_{local}/\\sigma_{global}}$ = " + f" {str(np.round(candidates_std/candidates_outside_std,2))}\nfor {textClass}")
        plt.savefig(savePath+'constraint_humanReadablePCA_'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()
        
        # also plot the distributions of the constraint values  
        fig, ax = plt.subplots(figsize=(5,5))
        
        # choose sensible number of bins based on the range of values and std
        binSize = 1
        for j in range(len(uniqueClassesGlobal)):
            indClassOutside = np.where(np.array(class_global)==uniqueClassesGlobal[j])[0]
            plt.hist(values_global[indClassOutside],bins=np.arange(np.min(values_global[indClassOutside]),np.max(values_global[indClassOutside])+binSize,binSize),ec='black',color=colorWheel[np.where(np.array(classWheel)==uniqueClassesGlobal[j])[0][0]],alpha=0.5,density=True,ls='dashed',label=uniqueClassesGlobal[j]+' ('+str(int(numClassesGlobal[j]))+'), (outside)')
        for j in range(len(uniqueClasses)):
            indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
            # plot the histograms for each class separately and inside and outside the subtree separately
            # inside the subtree the mfc is full
            # outside the subtree the mfc is empty
            plt.hist(values_subtree[indClass],bins=np.arange(np.min(values_global[indClassOutside]),np.max(values_global[indClassOutside])+binSize,binSize),color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],alpha=0.5,density=True,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
        plt.xlabel(f'values for constraint: {constraintText}')
        plt.ylabel('density')
        plt.legend(frameon=False)
        plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma_{local}/\\sigma_{global}}$ = " + f" {str(np.round(candidates_std/candidates_outside_std,2))}\nfor {textClass}")
        plt.savefig(savePath+'constraint_humanReadablePCA_hist_'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()            
            
    # next the "local" ones
    candidateConstraintsLocal_explainedPercentage = []
    for i in range(len(candidateConstraintsLocal)):
        
        constraintText = ''
        for j in range(len(candidateConstraintsLocal[i])):
            if candidateConstraintsLocal[i][j]!=0:
                if candidateConstraintsLocal[i][j]>0:
                    constraintText = constraintText + ' + '
                else:
                    constraintText = constraintText + ' - '
                constraintText = constraintText + str(np.abs(int(candidateConstraintsLocal[i][j]))) + constraintNames[j]
        constraintText = constraintText[3:]
        
        ind_outside = np.linspace(0,len(v_counts_outside)-1,len(v_counts_subtree)).astype(int)
        values_subtree = np.zeros(len(v_counts_subtree))
        values_subtreeNorm = np.zeros(len(v_counts_subtree))
        values_outside = np.zeros(len(v_counts_subtree))
        values_global = np.zeros(len(v_counts_outside))
        indNonZero = np.where(candidateConstraintsLocal[i,:]!=0)[0]

        values_subtree = np.sum(candidateConstraintsLocal[i,:]*v_counts_subtree,axis=1)
        values_subtreeNorm = np.sum(candidateConstraintsLocal[i,:]*Xsc,axis=1)
        values_outside = np.sum(candidateConstraintsLocal[i,:]*v_counts_outside[ind_outside,:],axis=1)
        values_global = np.sum(candidateConstraintsLocal[i,:]*v_counts_outside,axis=1)
        
        for j in range(len(uniqueClasses)):
            indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
            
        constraintVarNorm = np.sum(np.square(candidateConstraintsLocal[i,:]))
        candidates_var = np.nanvar(values_subtreeNorm)/constraintVarNorm
        candidateConstraintsLocal_explainedPercentage.append(candidates_var/np.sum(explained))

            
        if len(indNonZero)>1: # there are more than 1 so we can plot in the following way
            # divide coefficients into two nearly equal groups based on number of indices to plot against each other
            # we already have indNonZero
            numInds = len(indNonZero)
            if numInds%2==0:
                group0 = indNonZero[:int(numInds/2)]
                group1 = indNonZero[int(numInds/2):]
            else:
                group0 = indNonZero[:int(numInds/2)+1]
                group1 = indNonZero[int(numInds/2)+1:]
            # now plot group1 against group0
            values0 = np.zeros(len(v_counts_subtree))
            group0Name = ''
            for j in range(len(group0)):
                values0 = values0 + candidateConstraintsLocal[i,group0[j]]*v_counts_subtree[:,group0[j]]
                if candidateConstraintsLocal[i,group0[j]]>0:
                    group0Name = group0Name + ' + '
                else:
                    group0Name = group0Name + ' - '
                group0Name = group0Name + str(np.abs(int(candidateConstraintsLocal[i,group0[j]]))) + constraintNames[group0[j]]
            group0Name = group0Name[3:]
            if len(species) < len(globalSpecies): # then we can also plot the outgroup
                values0b = np.zeros(len(v_counts_outside))
                group0bName = ''
                for j in range(len(group0)):
                    values0b = values0b + candidateConstraintsLocal[i,group0[j]]*v_counts_outside[:,group0[j]]
                    if candidateConstraintsLocal[i,group0[j]]>0:
                        group0bName = group0bName + ' + '
                    else:
                        group0bName = group0bName + ' - '
                    group0bName = group0bName + str(np.abs(int(candidateConstraintsLocal[i,group0[j]]))) + constraintNames[group0[j]]
                group0bName = group0bName[3:]
            values1 = np.zeros(len(v_counts_subtree))
            group1Name = ''
            if candidateConstraintsLocal[i,group1[0]]>0: # to make sure the first coefficient is positive and we aren't plotting negative values against positive ones
                prefactor = 1
            else:
                prefactor = -1
            for j in range(len(group1)):
                values1 = values1 + prefactor*candidateConstraintsLocal[i,group1[j]]*v_counts_subtree[:,group1[j]]
                if candidateConstraintsLocal[i,group1[j]]>0:
                    group1Name = group1Name + ' + '
                else:
                    group1Name = group1Name + ' - '
                group1Name = group1Name + str(np.abs(int(candidateConstraintsLocal[i,group1[j]]))) + constraintNames[group1[j]]
            group1Name = group1Name[3:]
            if len(species) < len(globalSpecies): # then we can also plot the outgroup
                values1b = np.zeros(len(v_counts_outside))
                group1bName = ''
                for j in range(len(group1)):
                    values1b = values1b + prefactor*candidateConstraintsLocal[i,group1[j]]*v_counts_outside[:,group1[j]]
                    if candidateConstraintsLocal[i,group1[j]]>0:
                        group1bName = group1bName + ' + '
                    else:
                        group1bName = group1bName + ' - '
                    group1bName = group1bName + str(np.abs(int(candidateConstraintsLocal[i,group1[j]]))) + constraintNames[group1[j]]
                group1bName = group1bName[3:]
            for j in range(len(uniqueClasses)):
                indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
                plt.plot(values0[indClass],values1[indClass],color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],marker=markerWheelLocal[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],linestyle='None',alpha=0.5,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
            if len(species) < len(globalSpecies):
                for j in range(len(uniqueClassesGlobal)):
                    indClassOutside = np.where(np.array(class_global)==uniqueClassesGlobal[j])[0]
                    plt.plot(values0b[indClassOutside],values1b[indClassOutside],color=colorWheel[np.where(np.array(classWheel)==uniqueClassesGlobal[j])[0][0]],marker=markerWheelLocal[np.where(np.array(classWheel)==uniqueClassesGlobal[j])[0][0]],linestyle='None',alpha=0.5,mfc='None',label=uniqueClassesGlobal[j]+' ('+str(int(numClassesGlobal[j]))+') (outside)')
            plt.xlabel(f'{group0Name}')
            plt.ylabel(f'{group1Name}')
        
            plt.legend(frameon=False)

            plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma^2/\\sigma_{total}^2}$ = " + f" {str(np.round(candidateConstraintsLocal_explainedPercentage[-1],2))}\nfor {textClass}")
            plt.savefig(savePath+'constraintLocal_humanReadablePCA_'+str(i)+'.png',dpi=300,bbox_inches='tight')
            plt.close()
        
        # also plot the distributions of the constraint values if not looking at the full tree
        if len(species) < len(globalSpecies):
            
            fig, ax = plt.subplots(figsize=(5,5))
            
            # choose sensible number of bins based on the range of values and std
            binSize = 1
            for j in range(len(uniqueClassesGlobal)):
                indClassOutside = np.where(np.array(class_global)==uniqueClassesGlobal[j])[0]
                plt.hist(values_global[indClassOutside],bins=np.arange(np.min(values_global[indClassOutside]),np.max(values_global[indClassOutside])+binSize,binSize),ec='black',color=colorWheel[np.where(np.array(classWheel)==uniqueClassesGlobal[j])[0][0]],alpha=0.5,density=True,ls='dashed',label=uniqueClassesGlobal[j]+' ('+str(int(numClassesGlobal[j]))+'), (outside)')
            for j in range(len(uniqueClasses)):
                indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
                # plot the histograms for each class separately and inside and outside the subtree separately
                # inside the subtree the mfc is full
                # outside the subtree the mfc is empty
                plt.hist(values_subtree[indClass],bins=np.arange(np.min(values_global[indClassOutside]),np.max(values_global[indClassOutside])+binSize,binSize),color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],alpha=0.5,density=True,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
            plt.xlabel(f'values for constraint: {constraintText}')
            plt.ylabel('density')
            plt.legend(frameon=False)
            plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma^2/\\sigma_{total}^2}$ = " + f" {str(np.round(candidateConstraintsLocal_explainedPercentage[-1],2))}\nfor {textClass}")
            plt.savefig(savePath+'constraintLocal_humanReadablePCA_hist_'+str(i)+'.png',dpi=300,bbox_inches='tight')
            plt.close()
                    
    # and then plot something like the loading for all local and global constraints
    allConstraints = np.vstack((candidateConstraints,candidateConstraintsLocal))
    constraints_std = np.hstack((np.array(candidateConstraints_std)/np.array(candidateConstraints_outside_std),candidateConstraintsLocal_explainedPercentage))
    globalOrLocal = ['global']*len(candidateConstraints) + ['local']*len(candidateConstraintsLocal)
    
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(allConstraints,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=np.arange(len(allConstraints)), labels=[])
        # put the colorbar below the plot
        cbar = fig.colorbar(im, ticks=np.arange(np.min([np.min(allConstraints),0]),np.max(allConstraints)+1), aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')
        plt.text(np.shape(allConstraints)[1],-0.5,'frac.',va='center',ha='left')
    elif constraintName == 'CTCF':
        xLabels = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                xLabels.append(hoxLetters[i]+str(j+1))
        plt.xticks(ticks=np.arange(len(toCheck)), labels=xLabels, rotation=45, ha='right',fontsize=fontSize-2)
        plt.yticks(ticks=np.arange(len(allConstraints)), labels=[])
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(allConstraints)[1],-1,'frac.',va='center',ha='left')
    else: # set each ytick up through the length of allConstraints
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(allConstraints)[1],-1,'frac.',va='center',ha='left')
    for i in range(len(allConstraints)):
        plt.text(np.shape(allConstraints)[1],i,str(np.round(constraints_std[i],3)),va='center',ha='left')
        plt.text(-1,i,globalOrLocal[i],va='center',ha='right')
    plt.title(f"human-readable version of PCA constraints",y=1.05)
    plt.savefig(savePath+'constraintsAll_humanReadablePCA.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    # make a little dataframe with this new information of the local, global, etc. constraints
    
    constraintColumns = toCheck + ['constraint_std','globalOrLocal']
    constraintData = np.hstack((allConstraints,np.reshape(constraints_std,(len(constraints_std),1)),np.reshape(globalOrLocal,(len(globalOrLocal),1))))
    saveData = pd.DataFrame(constraintData,columns=constraintColumns)
    saveData.to_csv(savePath+'constraintsAll_humanReadablePCA.csv',index=False)
    
                
#%% function for plotting the PC1 vs. PC2
    
def plotPC1PC2(Xpca,groupList,xSubgroup,subgroup,savePath,explainedPercentage,rank,constraintName):
    
    # 2D plot of PC2 vs. PC1
    ind0 = 0
    ind1 = 1
    fig, ax = plt.subplots(figsize=(5,5))
    if groupList == 0:
        plt.plot(Xpca[:,ind0],Xpca[:,ind1],linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)#,label=f"mammals ({len(Xpca[xClass=='Mammalia',ind0])})")
    else:
        for i in range(len(groupList)):
            plt.plot(Xpca[pd.Series(xSubgroup)==groupList[i],ind0],Xpca[pd.Series(xSubgroup)==groupList[i],ind1],linestyle='None',marker=markerWheel[i],color=colorWheel[i],alpha=0.5,label=f"{groupList[i]} ({len(Xpca[pd.Series(xSubgroup)==groupList[i],ind0])})")
        plt.legend(frameon=False)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.ylim(-np.nanmax(np.abs(Xpca[:,ind0])),np.nanmax(np.abs(Xpca[:,ind0])))
    plt.title(f"PCA of {constraintName} data, rank ~ {rank}")
    plt.savefig(savePath+'pca.png',dpi=300,bbox_inches='tight')
    plt.close()
    
#%% plot loading of each feature on each PC
    
def plotLoading(loadings,explainedPercentage,savePath,rank,constraintName,toCheck):
    
    # 2D plot of loadings
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(loadings,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=[0,1,2,3,4], labels=['PC1*','PC2*','PC3*','PC4*','PC5*'])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
        plt.text(np.shape(loadings)[1],-0.5,'frac.',va='center',ha='left')
    elif constraintName == 'CTCF':
        # make x labels that go from A1 to A13, then B1 to B13, etc.
        xLabels = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                xLabels.append(hoxLetters[i]+str(j+1))
        plt.xticks(ticks=np.arange(len(xLabels)), labels=xLabels, ha='center',fontsize=fontSize-2, rotation=45)
        plt.yticks(ticks=np.arange(len(explainedPercentage)), labels=['PC'+str(i+1) for i in range(len(explainedPercentage))])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')
        plt.text(np.shape(loadings)[1],-1,'frac.',va='center',ha='left')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(explainedPercentage)), labels=['PC'+str(i+1) for i in range(len(explainedPercentage))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(loadings)[1],-1,'frac.',va='center',ha='left')
    for i in range(len(explainedPercentage)):
        plt.text(np.shape(loadings)[1],i,str(np.round(explainedPercentage[i],3)),va='center',ha='left')
    plt.title(f"PCA loading of {constraintName} data, rank ~ {rank}",y=1.05)
    plt.savefig(savePath+'loading.png',dpi=300,bbox_inches='tight')
    plt.close()
    
#%% plot the explained variance of each PC

def plotVariance(var,savePath,constraintName,toCheck):
    
    # 2D plot of variance (3 rows: individual, cumsum forward, cumsum reverse)
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(3,3))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(12,3))
        # set the aspect ratio to be 4 times the height of the plot
        ax.set_aspect(4)
    im = plt.imshow(var,cmap=colorMapVariance, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=[0,1,2], labels=['individual','cum. sum.','cum. sum.\n(reverse)'],rotation=0)
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
    elif constraintName == 'CTCF':
        # make x labels that go from A1 to A13, then B1 to B13, etc.
        xLabels = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                xLabels.append(hoxLetters[i]+str(j+1))
        plt.xticks(ticks=np.arange(len(xLabels)), labels=xLabels, ha='center',fontsize=fontSize-2, rotation=45)
        plt.yticks(ticks=[0,1,2], labels=['individual','cum. sum.','cum. sum. (reverse)'],rotation=30)
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=[0,1,2], labels=['individual','cum. sum.','cum. sum. (reverse)'],rotation=30)
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
    plt.title("$\mathregular{\\sigma_{local}/\\sigma_{global}}$" f" of {constraintName} data",y=1.05)
    plt.savefig(savePath+'var.png',dpi=300,bbox_inches='tight')
    plt.close()


#%% analyzeConstraints function

def analyzeConstraints(X,Xpca,species,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,constraints,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings):

    ind = np.where(constraints==1)[0]
    
    constraint = np.zeros((len(ind),len(toCheck)))
    varianceRatio = np.zeros(len(ind))
    constantValue = np.zeros(len(ind))
    
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    
    # multiply and sum loadings by globalXsc to get the global PC1 for any size of loadings and globalXsc
    globalPC1 = np.nansum(loadings[0,:]*globalXsc,axis=1)
    globalPC1Filtered = np.delete(globalPC1, indRemove, axis=0)
    
    for i in range(len(ind)):
        
        # normalize the loadings by the global standard deviation
        pre = loadings[ind[i],:]/globalStd
        # remove infs and replace with nans
        pre[np.isinf(pre)] = np.nan
        # normalize the constraint values by the maximum absolute value
        constraint[i,:] = pre/np.nanmax(np.abs(pre))
        constantValue[i] = np.nansum(np.nanmean(X.values,axis=0)*constraint[i,:])
        varianceRatio[i] = explained[ind[i]]/globalCov[ind[i]]
        # temp = loadings[ind[i],0]*globalXsc[:,0]+loadings[ind[i],1]*globalXsc[:,1]+loadings[ind[i],2]*globalXsc[:,2]+loadings[ind[i],3]*globalXsc[:,3]+loadings[ind[i],4]*globalXsc[:,4]
        temp = np.nansum(loadings[ind[i],:]*globalXsc,axis=1)
        tempFiltered = np.delete(temp, indRemove, axis=0)
        
        # 2D plot of the constrained PC components vs. PC1 and compared with the global data
        ind0 = 0
        ind1 = ind[i]
        fig, ax = plt.subplots(figsize=(5,5))
        plt.plot(globalPC1Filtered,tempFiltered,linestyle='None',marker='x',color='grey',alpha=0.5,label=f"global ({len(globalPC1Filtered)})")
        if groupList == 0:
            plt.plot(Xpca[:,ind0],Xpca[:,ind1],linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)#,label=f"mammals ({len(Xpca[xClass=='Mammalia',ind0])})")
        else:
            for j in range(len(groupList)):
                plt.plot(Xpca[pd.Series(xSubgroup)==groupList[j],ind0],Xpca[pd.Series(xSubgroup)==groupList[j],ind1],linestyle='None',marker=markerWheel[j],color=colorWheel[j],alpha=0.5,label=f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])})")
            plt.legend(frameon=False)
        plt.xlabel('PC1')
        plt.ylabel('PC'+str(ind[i]+1))
        # plt.ylim(-np.nanmax(np.abs(Xpca[:,ind0])),np.nanmax(np.abs(Xpca[:,ind0])))
        plt.ylim(-np.nanmax(np.abs(tempFiltered)),np.nanmax(np.abs(tempFiltered)))
        # plt.title(f"possible constraint ({np.round(100*explainedPercentage[ind[i]],2)}%):\n{np.round(loadings[ind[i],0],2)}C + {np.round(loadings[ind[i],1],2)}T + {np.round(loadings[ind[i],2],2)}L + {np.round(loadings[ind[i],3],2)}S + {np.round(loadings[ind[i],4],2)}Ca ~ constant")
        if constraintName == 'vertebral':
            plt.title(f"possible constraint ({np.round(100*explainedPercentage[ind[i]],3)}%): "+"$\mathregular{\\sigma_{local}/\\sigma_{global}}$"+f" = {np.round(explained[ind[i]]/globalCov[ind[i]],3)}\n{np.round(constraint[i,0],2)}C + {np.round(constraint[i,1],2)}T + {np.round(constraint[i,2],2)}L + {np.round(constraint[i,3],2)}S + {np.round(constraint[i,4],2)}Ca ~ {np.round(constantValue[i],2)}")
        else:
            plt.title(f"possible constraint ({np.round(100*explainedPercentage[ind[i]],3)}%): "+"$\mathregular{\\sigma_{local}/\\sigma_{global}}$"+f" = {np.round(explained[ind[i]]/globalCov[ind[i]],3)}")
        plt.savefig(savePath+'subtree_possibleConstraint'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()

    dfConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfConstraint[toCheck[i]] = constraint[:,i]
    dfConstraint['constantValue'] = constantValue
    dfConstraint['varianceRatio'] = varianceRatio
    dfConstraint.to_csv(savePath+'constraints.csv',index=False,float_format="%.10g")
    
    # 2D plot of constraints (similar to loading plot)
    plotMatrix = constraint # if we were to include the constant as well: np.hstack((constraint,np.reshape(constantValue,(len(constantValue),1))))
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(plotMatrix,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-0.5,"$\mathregular{\\sigma_{loc.}/\\sigma_{glob.}}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"constraints extracted from {constraintName} data",y=1.05)
    else:
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])}), "
        plt.title(f"constraints extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'constraints.png',dpi=300,bbox_inches='tight')
    plt.close()
    
#%% analyzeConstraints function (local)

def analyzeConstraintsLocal(X,Xpca,species,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,constraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings):

    ind = np.where(constraintsLocal==1)[0]
    
    constraint = np.zeros((len(ind),len(toCheck)))
    varianceRatio = np.zeros(len(ind))
    ind0 = 0
        
    for i in range(len(ind)):
        constraint[i,:] = loadings[ind[i],:]
        varianceRatio[i] = explainedPercentage[ind[i]]

    dfConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfConstraint[toCheck[i]] = constraint[:,i]
    dfConstraint['varianceRatio'] = varianceRatio
    dfConstraint.to_csv(savePath+'constraintsLocal.csv',index=False,float_format="%.10g")
    
    # 2D plot of constraints (similar to loading plot)
    plotMatrix = constraint # if we were to include the constant as well: np.hstack((constraint,np.reshape(constantValue,(len(constantValue),1))))
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(plotMatrix,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-0.5,"$\mathregular{\\sigma_{loc.}/\\sigma_{glob.}}$",va='center',ha='center')
    elif constraintName == 'CTCF':
        # make x labels that go from A1 to A13, then B1 to B13, etc.
        xLabels = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                xLabels.append(hoxLetters[i]+str(j+1))
        plt.xticks(ticks=np.arange(len(xLabels)), labels=xLabels, ha='center',fontsize=fontSize-2, rotation=45)
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"constraints extracted from {constraintName} data",y=1.05)
    else:
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])}), "
        plt.title(f"local constraints extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'constraintsLocal.png',dpi=300,bbox_inches='tight')
    plt.close()

#%% analyzeNonConstraints function (local)

def analyzeNonConstraintsLocal(X,Xpca,species,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,nonConstraintsLocal,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings):

    ind = np.where(nonConstraintsLocal==1)[0]
    
    nonConstraint = np.zeros((len(ind),len(toCheck)))
    varianceRatio = np.zeros(len(ind))
    ind0 = 0    
        
    for i in range(len(ind)):
        nonConstraint[i,:] = loadings[ind[i],:]
        varianceRatio[i] = explainedPercentage[ind[i]]

    dfNonConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfNonConstraint[toCheck[i]] = nonConstraint[:,i]
    dfNonConstraint['varianceRatio'] = varianceRatio
    dfNonConstraint.to_csv(savePath+'nonConstraintsLocal.csv',index=False,float_format="%.10g")
    
    # 2D plot of non constraints (similar to loading plot)
    plotMatrix = nonConstraint # if we were to include the constant as well: np.hstack((constraint,np.reshape(constantValue,(len(constantValue),1))))
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(plotMatrix,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-0.5,"$\mathregular{\\sigma_{loc.}/\\sigma_{glob.}}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"non-constraints extracted from {constraintName} data",y=1.05)
    else:
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])}), "
        plt.title(f"local non-constraints extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'nonConstraintsLocal.png',dpi=300,bbox_inches='tight')
    plt.close()
    
#%% analyzeNonConstraints function

def analyzeNonConstraints(X,Xpca,species,globalXsc,globalSpecies,groupList,xSubgroup,subgroup,savePath,explained,explainedPercentage,rank,nonConstraints,toCheck,maxPCA,globalStd,globalCov,constraintName,loadings):

    ind = np.where(nonConstraints==1)[0]
    
    nonConstraint = np.zeros((len(ind),len(toCheck)))
    varianceRatio = np.zeros(len(ind))
    constantValue = np.zeros(len(ind))
    
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    
    for i in range(len(ind)):
        
        # normalize the loadings by the global standard deviation
        pre = loadings[ind[i],:]/globalStd
        # remove infs and replace with nans
        pre[np.isinf(pre)] = np.nan
        # normalize the constraint values by the maximum absolute value
        nonConstraint[i,:] = pre/np.nanmax(np.abs(pre))
        constantValue[i] = np.nansum(np.nanmean(X.values,axis=0)*nonConstraint[i,:])
        varianceRatio[i] = explained[ind[i]]/globalCov[ind[i]]
        temp = np.nansum(loadings[ind[i],:]*globalXsc,axis=1)
        tempFiltered = np.delete(temp, indRemove, axis=0)
        # multiply and sum loadings by globalXsc to get the global PC1 for any size of loadings and globalXsc
        if ind[i] == np.shape(Xpca)[1]-1: # sometimes the last PC is strangely the one with the most global variation!
            indPC = np.shape(Xpca)[1]-2
        else:
            indPC = ind[i]+1
        globalPC = np.nansum(loadings[indPC,:]*globalXsc,axis=1)
        globalPCFiltered = np.delete(globalPC, indRemove, axis=0)
        
        # 2D plot of the non-constrained PC components vs. another PC component with global sampling
        ind0 = indPC
        ind1 = ind[i]
        fig, ax = plt.subplots(figsize=(5,5))
        plt.plot(globalPCFiltered,tempFiltered,linestyle='None',marker='x',color='grey',alpha=0.5,label=f"global ({len(globalPCFiltered)})")
        if groupList == 0:
            plt.plot(Xpca[:,ind0],Xpca[:,ind1],linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)#,label=f"mammals ({len(Xpca[xClass=='Mammalia',ind0])})")
        else:
            for j in range(len(groupList)):
                plt.plot(Xpca[pd.Series(xSubgroup)==groupList[j],ind0],Xpca[pd.Series(xSubgroup)==groupList[j],ind1],linestyle='None',marker=markerWheel[j],color=colorWheel[j],alpha=0.5,label=f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])})")
            plt.legend(frameon=False)
        plt.xlabel('PC'+str(ind[i]+2))
        plt.ylabel('PC'+str(ind[i]+1))
        plt.ylim(-np.nanmax(np.abs(Xpca[:,ind0])),np.nanmax(np.abs(Xpca[:,ind0])))
        # plt.ylim(-np.nanmax(np.abs(tempFiltered)),np.nanmax(np.abs(tempFiltered)))
        # plt.title(f"possible constraint ({np.round(100*explainedPercentage[ind[i]],2)}%):\n{np.round(loadings[ind[i],0],2)}C + {np.round(loadings[ind[i],1],2)}T + {np.round(loadings[ind[i],2],2)}L + {np.round(loadings[ind[i],3],2)}S + {np.round(loadings[ind[i],4],2)}Ca ~ constant")
        if constraintName == 'vertebral':
            plt.title(f"possible plasticity ({np.round(100*explainedPercentage[ind[i]],3)}%): "+"$\mathregular{\\sigma_{local}/\\sigma_{global}}$"+f" = {np.round(explained[ind[i]]/globalCov[ind[i]],3)}\n{np.round(nonConstraint[i,0],2)}C + {np.round(nonConstraint[i,1],2)}T + {np.round(nonConstraint[i,2],2)}L + {np.round(nonConstraint[i,3],2)}S + {np.round(nonConstraint[i,4],2)}Ca")
        else:
            plt.title(f"possible plasticity ({np.round(100*explainedPercentage[ind[i]],3)}%): "+"$\mathregular{\\sigma_{local}/\\sigma_{global}}$"+f" = {np.round(explained[ind[i]]/globalCov[ind[i]],3)}")
        plt.savefig(savePath+'subtree_possiblePlasticity'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()

    dfNonConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfNonConstraint[toCheck[i]] = nonConstraint[:,i]
    dfNonConstraint['constantValue'] = constantValue
    dfNonConstraint['varianceRatio'] = varianceRatio
    dfNonConstraint.to_csv(savePath+'plasticity.csv',index=False,float_format="%.10g")
    
    # 2D plot of constraints (similar to loading plot but now normalized and includes the constant value)
    plotMatrix = nonConstraint
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(5,5))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(14,5))
        # set the aspect ratio to be 5 times the height of the plot
        ax.set_aspect(5)
    im = plt.imshow(plotMatrix,cmap=colorMap, interpolation='nearest')
    if constraintName == 'vertebral':
        plt.xticks(ticks=[0,1,2,3,4], labels=['C','T','L','S','Ca'])
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # put the colorbar below the plot
        plt.colorbar(aspect=20, pad=0.13, shrink=0.6,orientation='horizontal')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-0.5,"$\mathregular{\\sigma_{loc.}/\\sigma_{glob.}}$",va='center',ha='center')
    elif constraintName == 'CTCF':
        xLabels = []
        hoxLetters = ['A','B','C','D']
        for i in range(len(hoxLetters)):
            for j in range(13):
                xLabels.append(hoxLetters[i]+str(j+1))
        plt.xticks(ticks=np.arange(len(xLabels)), labels=xLabels, ha='center',fontsize=fontSize-2, rotation=45)
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['P'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"plasticity extracted from {constraintName} data",y=1.05)
    else:
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({len(Xpca[pd.Series(xSubgroup)==groupList[j],ind0])}), "
        plt.title(f"plasticity extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'plasticity.png',dpi=300,bbox_inches='tight')
    plt.close()
    