# individual functions for the vertebral paper

#%% libraries

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

#%% function that analyzes individual vertebrae given a list of species and their features

def constraintIndividual(df,tree,species,globalSpecies,constraintName,toCheck,subgroup,groupList,savePath,constraintRatio,plasticityRatio,constraintRatioHumanReadable,constraintThresholdLocal):
    
    # get the specific values we want
    X = df.loc[df['Species'].isin(species),toCheck]
    if groupList == 0:
        xSubgroup = 0
    else:
        xSubgroup = df.loc[df['Species'].isin(species),subgroup].squeeze().tolist()
    xSpecies = df.loc[df['Species'].isin(species),'Species'].squeeze().tolist()
        
    totalVar = np.trace(np.cov(X.values.T)) # the local total variance
        
    # get the indices of the this subtree's species in order to drop from the global list
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    Xglobal = df.loc[:,toCheck]
    Xglobal = Xglobal.drop(indRemove)
    
    varLocal = np.nanvar(X.values,axis=0)
    varTotalLocal = np.nanvar(X.values,axis=0)/totalVar
    
    # get the variation outside this subtree
    globalVar = np.nanvar(Xglobal.loc[:,pd.Index(toCheck)],axis=0)
    globalVarCumsumForward = np.nanvar(np.cumsum(Xglobal.loc[:,pd.Index(toCheck)],axis=1),axis=0)
    globalVarCumsumBackward = np.nanvar(np.fliplr(np.cumsum(np.fliplr(Xglobal.loc[:,pd.Index(toCheck)]),axis=1)),axis=0)
    
    # calculate the local variation over global for each individual toCheck value as well as the forward and reverse cumsums
    varLocalGlobal = np.nanvar(X.values,axis=0)/globalVar
    varLocalGlobalCumsumForward = np.nanvar(np.cumsum(X.values,axis=1),axis=0)/globalVarCumsumForward
    varLocalGlobalCumsumBackward = np.nanvar(np.fliplr(np.cumsum(np.fliplr(X.values),axis=1)),axis=0)/globalVarCumsumBackward
    
    # combine
    var = np.vstack((varLocal,varTotalLocal,varLocalGlobal,varLocalGlobalCumsumForward,varLocalGlobalCumsumBackward))
    # replace infs with nans
    var[var == np.inf] = np.nan
    var[var == -np.inf] = np.nan
    
    # save the data
    np.savetxt(savePath+'var.csv',var,delimiter=',') # save variation data
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
    
    # just do the constraints here as well (global)
    constraints = np.zeros((3,len(toCheck)))
    nonConstraints = np.zeros((3,len(toCheck)))
    indGlobal = ([2,3,4])
    for i in range(3):
        for j in range(np.shape(var)[1]):
            if var[indGlobal[i],j] < 1/constraintRatio:
                constraints[i,j] = 1
            if var[indGlobal[i],j] > plasticityRatio:
                nonConstraints[i,j] = 1    
                
    # now do the local version
    constraintsLocal = np.zeros(len(toCheck))
    nonConstraintsLocal = np.zeros(len(toCheck))
    for i in range(len(toCheck)):
        if var[1,i] < 1/constraintRatioHumanReadable:
            constraintsLocal[i] = 1
        if var[1,i] > constraintThresholdLocal:
            nonConstraintsLocal[i] = 1

    return(X,var,xSubgroup,constraints,nonConstraints,constraintsLocal,nonConstraintsLocal)



#%% analyzeConstraints function for individual constraints

def analyzeConstraintsIndividual(df,var,species,globalSpecies,groupList,xSubgroup,subgroup,savePath,constraints,toCheck,constraintName):
    
    # reduce to the values to check only and then the species in the subtree
    indSpecies = [i for i, s in enumerate(df['Species']) if s in species]
    X = df.iloc[indSpecies,:]
    X = X[toCheck]
    X = X.reset_index(drop=True)
    
    # get the indices of the this subtree's species in order to drop from the global list
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    Xglobal = df.loc[:,toCheck]
    Xglobal = Xglobal.drop(indRemove)
    Xglobal.reset_index(drop=True,inplace=True)
    
    indShift = 2 # after the first two are the ones I was looking at before 12 september 2023
    
    saveNames = (['individual','cumsum','cumsumReverse'])
    for q in range(3):
        varianceRatio = []
        ind = np.where(constraints[q,:]==1)[0]
        if len(ind) == 0:
            continue
        constraintValues = np.zeros((len(ind),len(toCheck)))
        constraint = np.zeros((len(ind),len(toCheck)))
        for i in range(len(ind)):
            constraintValues[i,ind[i]] = np.nanmean(X.values[:,ind[i]])
            constraint[i,ind[i]] = 1
            varianceRatio.append(var[q+indShift,ind[i]])
        
        # save the constraint values
        dfConstraintValues = pd.DataFrame()
        for i in range(len(toCheck)):
            dfConstraintValues[toCheck[i]] = constraintValues[:,i]
        dfConstraintValues['varianceRatio'] = varianceRatio
        dfConstraintValues.to_csv(savePath+'constraintValues_'+saveNames[q]+'.csv',index=False,float_format="%.10g")
        
        # save the constraint coefficients themselves
        dfConstraint = pd.DataFrame()
        for i in range(len(toCheck)):
            dfConstraint[toCheck[i]] = constraint[:,i]
        dfConstraint['varianceRatio'] = varianceRatio
        dfConstraint.to_csv(savePath+'constraints_'+saveNames[q]+'.csv',index=False,float_format="%.10g")
        
        # 2D plot of constraints (similar to loading plot)
        plotMatrix = constraint[indShift:,:] # if we were to include the constant as well: np.hstack((constraint,np.reshape(constantValue,(len(constantValue),1))))
        if constraintName == 'vertebral':
            fig, ax = plt.subplots(figsize=(3,3))
        elif 'CTCF' in constraintName:
            fig, ax = plt.subplots(figsize=(12,3))
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
            plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
            # plot all the toCheck features on the x axis tilted at an angle 45
            plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
            # put the colorbar beside the plot
            plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
            plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
        for i in range(len(varianceRatio)):
            plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
        if groupList == 0:
            plt.title(f"individual constraints extracted from {constraintName} data",y=1.05)
        else:
            # get the number of xSubgroup in each element of groupList
            numGroup = []
            for j in range(len(groupList)):
                numGroup.append(len([i for i in xSubgroup if i == groupList[j]]))
            titleText = ''
            for j in range(len(groupList)):
                titleText = titleText + f"{groupList[j]} ({numGroup[j]}), "
            plt.title(f"individual constraints extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

        plt.savefig(savePath+'constraints_'+saveNames[q]+'.png',dpi=300,bbox_inches='tight')
        plt.close()
        
        # take the same number of values as in species from the global data, randomly chosen
        if len(Xglobal) > len(indSpecies):
            indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=False)
        else:
            indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=True)
            
        for i in range(len(ind)):
            globalValues = Xglobal.iloc[indGlobalNotInSubtree,ind[i]].values
            localValues = X.iloc[:,ind[i]].values
            fig, ax = plt.subplots(figsize=(3,3))
            plt.plot(globalValues,localValues,linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)
            plt.xlabel('global')
            plt.ylabel('subtree')
            plt.xlim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            plt.ylim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            if constraintName == 'vertebral':
                plt.title('global vs. subtree values of '+toCheck[ind[i]])
            else:
                plt.title('global vs. subtree values of\n'+toCheck[ind[i]])
            plt.savefig(savePath+'subtreeVsGlobal_constraint_'+saveNames[q]+'_'+toCheck[ind[i]]+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
#%% local version

def analyzeConstraintsIndividualLocal(df,var,species,globalSpecies,groupList,xSubgroup,subgroup,savePath,constraints,toCheck,constraintName):
    
    # reduce to the values to check only and then the species in the subtree
    indSpecies = [i for i, s in enumerate(df['Species']) if s in species]
    X = df.iloc[indSpecies,:]
    X = X[toCheck]
    X = X.reset_index(drop=True)
    
    # get the indices of the this subtree's species in order to drop from the global list
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    Xglobal = df.loc[:,toCheck]
    Xglobal = Xglobal.drop(indRemove)
    Xglobal.reset_index(drop=True,inplace=True)
    
    indConstraint = np.where(constraints==1)[0]
    indConstraintKeep = []
    variance = []
    varianceRatio = []
    
    # some checks and a plot
    for q in range(len(indConstraint)):
        if (np.sum(np.abs(X.values[:,indConstraint[q]])) == 0):# | (len(indConstraint[q]) == 0): # first check that the values are not simply all zero or that there aren't any!
            continue
        else:
            indConstraintKeep.append(indConstraint[q])
            variance.append(var[0,indConstraint[q]])
            varianceRatio.append(var[1,indConstraint[q]])
            
            # take the same number of values as in species from the global data, randomly chosen
            if len(Xglobal) > len(indSpecies):
                indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=False)
            elif len(Xglobal) == 0:
                continue
            else:
                indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=True)
                
            # plot
            globalValues = Xglobal.iloc[indGlobalNotInSubtree,indConstraint[q]].values
            localValues = X.iloc[:,indConstraint[q]].values
            fig, ax = plt.subplots(figsize=(3,3))
            plt.plot(globalValues,localValues,linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)
            plt.xlabel('global')
            plt.ylabel('subtree')
            plt.xlim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            plt.ylim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            if constraintName == 'vertebral':
                plt.title('global vs. subtree values of '+toCheck[indConstraint[q]])
            else:
                plt.title('global vs. subtree values of\n'+toCheck[indConstraint[q]])
            plt.savefig(savePath+'subtreeVsGlobal_constraintIndividualLocal'+'_'+toCheck[indConstraint[q]]+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
    # save this information
    constraintsKeep = np.zeros((len(indConstraintKeep),len(toCheck)))
    for i in range(len(indConstraintKeep)):
        constraintsKeep[i,indConstraintKeep[i]] = 1

    # save the constraint coefficients themselves
    dfConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfConstraint[toCheck[i]] = constraintsKeep[:,i]
    dfConstraint['variance'] = variance
    dfConstraint['varianceRatio'] = varianceRatio
    dfConstraint.to_csv(savePath+'constraintsIndividualLocal.csv',index=False,float_format="%.10g")
    
    # 2D plot of constraints (similar to loading plot)
    plotMatrix = constraintsKeep # if we were to include the constant as well: np.hstack((constraintsKeep,np.reshape(constantValue,(len(constantValue),1))))
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(3,3))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(12,3))
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
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma^2/\\sigma_{total}^2}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma^2/\\sigma_{total}^2}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"individual constraints extracted from {constraintName} data",y=1.05)
    else:
        # get the number of xSubgroup in each element of groupList
        numGroup = []
        for j in range(len(groupList)):
            numGroup.append(len([i for i in xSubgroup if i == groupList[j]]))
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({numGroup[j]}), "
        plt.title(f"individual local constraints extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'constraintsIndividualLocal.png',dpi=300,bbox_inches='tight')
    plt.close()
        
        
#%% analyzeConstraints function for individual constraints


def analyzeNonConstraintsIndividual(df,var,species,globalSpecies,groupList,xSubgroup,subgroup,savePath,nonconstraints,toCheck,constraintName):
    
    # reduce to the values to check only and then the species in the subtree
    indSpecies = [i for i, s in enumerate(df['Species']) if s in species]
    X = df.iloc[indSpecies,:]
    X = X[toCheck]
    X = X.reset_index(drop=True)
    
    # get the indices of the this subtree's species in order to drop from the global list
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    Xglobal = df.loc[:,toCheck]
    Xglobal = Xglobal.drop(indRemove)
    Xglobal.reset_index(drop=True,inplace=True)
    
    indShift = 2 # after the first two are the ones I was looking at before 12 september 2023
    
    saveNames = (['individual','cumsum','cumsumReverse'])
    for q in range(3):
        varianceRatio = []
        ind = np.where(nonconstraints[q,:]==1)[0]
        if len(ind) == 0:
            continue
        nonconstraintValues = np.zeros((len(ind),len(toCheck)))
        nonconstraint = np.zeros((len(ind),len(toCheck)))
        for i in range(len(ind)):
            nonconstraintValues[i,ind[i]] = np.nanmean(X.values[:,ind[i]])
            nonconstraint[i,ind[i]] = 1
            varianceRatio.append(var[q+indShift,ind[i]])
        
        # save the non-constraint values
        dfNonConstraintValues = pd.DataFrame()
        for i in range(len(toCheck)):
            dfNonConstraintValues[toCheck[i]] = nonconstraintValues[:,i]
        dfNonConstraintValues['varianceRatio'] = varianceRatio
        dfNonConstraintValues.to_csv(savePath+'plasticityValues_'+saveNames[q]+'.csv',index=False,float_format="%.10g")
        
        # save the plasticity coefficients themselves
        dfNonConstraint = pd.DataFrame()
        for i in range(len(toCheck)):
            dfNonConstraint[toCheck[i]] = nonconstraint[:,i]
        dfNonConstraint['varianceRatio'] = varianceRatio
        dfNonConstraint.to_csv(savePath+'plasticity_'+saveNames[q]+'.csv',index=False,float_format="%.10g")
        
        # 2D plot of constraints (similar to loading plot)
        plotMatrix = nonconstraint[indShift:,:] # if we were to include the constant as well: np.hstack((constraint,np.reshape(constantValue,(len(constantValue),1))))
        if constraintName == 'vertebral':
            fig, ax = plt.subplots(figsize=(3,3))
        elif 'CTCF' in constraintName:
            fig, ax = plt.subplots(figsize=(12,3))
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
            plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
            # plot all the toCheck features on the x axis tilted at an angle 45
            plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
            # put the colorbar beside the plot
            plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
            plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma_{local}/\\sigma_{global}}$",va='center',ha='center')
        for i in range(len(varianceRatio)):
            plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
        if groupList == 0:
            plt.title(f"individual plasticity extracted from {constraintName} data",y=1.05)
        else:
            # get the number of xSubgroup in each element of groupList
            numGroup = []
            for j in range(len(groupList)):
                numGroup.append(len([i for i in xSubgroup if i == groupList[j]]))
            titleText = ''
            for j in range(len(groupList)):
                titleText = titleText + f"{groupList[j]} ({numGroup[j]}), "
            plt.title(f"individual plasticity extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

        plt.savefig(savePath+'plasticity_'+saveNames[q]+'.png',dpi=300,bbox_inches='tight')
        plt.close()
        
        # take the same number of values as in species from the global data, randomly chosen
        if len(Xglobal) > len(indSpecies):
            indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=False)
        else:
            indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=True)
        
        # also make a plot for each individual constraint the values for the species in this group and outside this group
        for i in range(len(ind)):
            globalValues = Xglobal.iloc[indGlobalNotInSubtree,ind[i]].values
            localValues = X.iloc[:,ind[i]].values
            fig, ax = plt.subplots(figsize=(3,3))
            plt.plot(globalValues,localValues,linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)
            plt.xlabel('global')
            plt.ylabel('subtree')
            plt.xlim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            plt.ylim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            plt.title('global vs. subtree values of '+toCheck[ind[i]])
            plt.savefig(savePath+'subtreeVsGlobal_nonConstraint_'+saveNames[q]+'_'+toCheck[ind[i]]+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
#%% local version

def analyzeNonConstraintsIndividualLocal(df,var,species,globalSpecies,groupList,xSubgroup,subgroup,savePath,nonConstraints,toCheck,constraintName):
    
    # reduce to the values to check only and then the species in the subtree
    indSpecies = [i for i, s in enumerate(df['Species']) if s in species]
    X = df.iloc[indSpecies,:]
    X = X[toCheck]
    X = X.reset_index(drop=True)
    
    # get the indices of the this subtree's species in order to drop from the global list
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    Xglobal = df.loc[:,toCheck]
    Xglobal = Xglobal.drop(indRemove)
    Xglobal.reset_index(drop=True,inplace=True)
    
    indNonConstraint = np.where(nonConstraints==1)[0]
    indNonConstraintKeep = []
    variance = []
    varianceRatio = []
    
    # some checks and a plot
    for q in range(len(indNonConstraint)):
        if (np.sum(np.abs(X.values[:,indNonConstraint[q]])) == 0):# | (len(indNonConstraint[q]) == 0): # first check that the values are not simply all zero or that there aren't any!
            continue
        else:
            indNonConstraintKeep.append(indNonConstraint[q])
            variance.append(var[0,indNonConstraint[q]])
            varianceRatio.append(var[1,indNonConstraint[q]])
            
            # take the same number of values as in species from the global data, randomly chosen
            if len(Xglobal) > len(indSpecies):
                indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=False)
            elif len(Xglobal) == 0:
                continue
            else:
                indGlobalNotInSubtree = np.random.choice(np.arange(len(Xglobal)),len(indSpecies),replace=True)
                
            # plot
            globalValues = Xglobal.iloc[indGlobalNotInSubtree,indNonConstraint[q]].values
            localValues = X.iloc[:,indNonConstraint[q]].values
            fig, ax = plt.subplots(figsize=(3,3))
            plt.plot(globalValues,localValues,linestyle='None',marker='o',color=colorWheel[0],alpha=0.5)
            plt.xlabel('global')
            plt.ylabel('subtree')
            plt.xlim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            plt.ylim(np.min([-1,0.9*np.nanmin(globalValues)]),np.max([1.1*np.nanmax(localValues),1.1*np.nanmax(globalValues)]))
            if constraintName == 'vertebral':
                plt.title('global vs. subtree values of '+toCheck[indNonConstraint[q]])
            else:
                plt.title('global vs. subtree values of\n'+toCheck[indNonConstraint[q]])
            plt.savefig(savePath+'subtreeVsGlobal_plasticityIndividualLocal'+'_'+toCheck[indNonConstraint[q]]+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
    # save this information
    nonConstraintsKeep = np.zeros((len(indNonConstraintKeep),len(toCheck)))
    for i in range(len(indNonConstraintKeep)):
        nonConstraintsKeep[i,indNonConstraintKeep[i]] = 1

    # save the constraint coefficients themselves
    dfNonConstraint = pd.DataFrame()
    for i in range(len(toCheck)):
        dfNonConstraint[toCheck[i]] = nonConstraintsKeep[:,i]
    dfNonConstraint['variance'] = variance
    dfNonConstraint['varianceRatio'] = varianceRatio
    dfNonConstraint.to_csv(savePath+'plasticityIndividualLocal.csv',index=False,float_format="%.10g")
    
    # 2D plot of constraints (similar to loading plot)
    plotMatrix = nonConstraintsKeep # if we were to include the constant as well: np.hstack((constraintsKeep,np.reshape(constantValue,(len(constantValue),1))))
    if constraintName == 'vertebral':
        fig, ax = plt.subplots(figsize=(3,3))
    elif 'CTCF' in constraintName:
        fig, ax = plt.subplots(figsize=(12,3))
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
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma^2/\\sigma_{total}^2}$",va='center',ha='center')
    else: # name each ytick PC1, PC2, etc. up through the length of explainedPercentage
        plt.yticks(ticks=np.arange(len(varianceRatio)), labels=['C'+str(i) for i in range(len(varianceRatio))])
        # plot all the toCheck features on the x axis tilted at an angle 45
        plt.xticks(ticks=np.arange(len(toCheck)), labels=toCheck, rotation=45, ha='right',fontsize=fontSize-2)
        # put the colorbar beside the plot
        plt.colorbar(aspect=20, pad=0.07, shrink=0.4,orientation='vertical')#, extend='both')
        plt.text(np.shape(plotMatrix)[1],-1.5,"$\mathregular{\\sigma^2/\\sigma_{total}^2}$",va='center',ha='center')
    for i in range(len(varianceRatio)):
        plt.text(np.shape(plotMatrix)[1],i,str(np.round(varianceRatio[i],3)),va='center',ha='left')
    if groupList == 0:
        plt.title(f"individual plasticities extracted from {constraintName} data",y=1.05)
    else:
        # get the number of xSubgroup in each element of groupList
        numGroup = []
        for j in range(len(groupList)):
            numGroup.append(len([i for i in xSubgroup if i == groupList[j]]))
        titleText = ''
        for j in range(len(groupList)):
            titleText = titleText + f"{groupList[j]} ({numGroup[j]}), "
        plt.title(f"individual local plasticities extracted from {constraintName} data\n"+titleText[:-2],y=1.05)

    plt.savefig(savePath+'plasticityIndividualLocal.png',dpi=300,bbox_inches='tight')
    plt.close()
        