# functions for the "human-readable" analysis or search for constraints

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

#%% function to make the readable matrix of constraints (this takes a lot of time so just do once!)

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
        
#%% testing for human-readable constraints
        
        
def constraintHumanReadable(v_counts_subtree,species,full_tree,globalSpecies,constraintRatioHumanReadable,plasticityRatioHumanReadable,constraintThresholdLocal,plasticityThresholdLocal,toCheck,cvec_unique,datadir,spanHumanReadable):

    # we want to also do something for the full tree (so when len(globalSpecies) = 0)
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    v_counts_outside = full_tree.loc[:,pd.Index(toCheck)].values
    if len(indRemove)<len(globalSpecies):
        v_counts_outside = np.delete(v_counts_outside, indRemove, axis=0)
    
    totalVar = np.trace(np.cov(v_counts_subtree.T))

    # calculate the 'interpretable' constrained values
    in_value_constrained=np.dot(cvec_unique,v_counts_subtree.T)
    # calculate the variance of the 'interpretable' constrained values
    in_value_constrained_std=np.std(in_value_constrained,axis=1)
    
    # if we don't normalize then the variance will depend on the magnitude of the coefficients...
    in_value_constrained_std_norm = np.sqrt(np.sum(cvec_unique**2,axis=1))
    in_value_constrained_std = in_value_constrained_std/in_value_constrained_std_norm
    
    # this is for a global constraint threshold, but I need to check it for each constraint "loading"
    # check each "loading"
    constrained_outside_std = []
    for i in range(len(cvec_unique)):
        constrained_outside_std.append(np.nanstd(np.sum(cvec_unique[i,:]*v_counts_outside,axis=1))/in_value_constrained_std_norm[i])
    constrained_outside_std = np.array(constrained_outside_std)
    
    if len(indRemove)<len(globalSpecies):
                    
        # reduce: need to satisfy either inside or outside the subtree
        indReduce = []
        for i in range(len(cvec_unique)):
            if (in_value_constrained_std[i]**2<totalVar*constraintThresholdLocal) | (in_value_constrained_std[i]**2<(constrained_outside_std[i]**2)/constraintRatioHumanReadable):
                indReduce.append(i)
        candidates = cvec_unique[indReduce,:]
        candidates_std = in_value_constrained_std[indReduce]
        candidates_outside_std = np.array(constrained_outside_std)[indReduce]

        # reduce to the linearly independent vectors, sorting by lowest candidates_std**2/totalVariance
        # first sort
        # now sort not just by the variance ratio but also by the "complexity" of the constraint (size of largest coefficient)
        complexity = np.max(np.abs(candidates),axis=1)
        candidates_std_sort_ind = np.lexsort((candidates_std**2/totalVar,complexity))
        # candidates_std_sort_ind = np.argsort(candidates_std/candidates_outside_std)
        candidates_std = candidates_std[candidates_std_sort_ind]
        candidates_outside_std = candidates_outside_std[candidates_std_sort_ind]
        candidates = candidates[candidates_std_sort_ind,:]
        # now reduce to the linearly independent vectors, up through the number of non-readable constraints found
        _, inds = sp.Matrix(candidates).T.rref()  # to check the rows you need to transpose!

        # constraints
        finalCandidates = candidates[inds,:]
        finalCandidates_std = np.ndarray.flatten(candidates_std[inds,None])
        finalCandidates_outside_std = np.ndarray.flatten(candidates_outside_std[inds,None])
        
        # starting from the simplest constraint, and the one that has the least variance ratio,
        # remove this constraints coefficients from the other constraints
        # if the variance ratio of the other constraints is not increased by more than a threshold, then remove the coefficients
        # otherwise see if adding the constraint coefficients improves the constraint
        # the rows are already sorted by variance ratio
        
        # if any constraints have zero standard deviation or nearly zero (var/totalVar < 0.01), 
        # then remove that coefficient from the rest of the constraints
        # but I can only do this in this way when there is only one coefficient
        indZeroVariance = np.where(finalCandidates_std**2/totalVar<0.01)[0]
        if len(indZeroVariance)>0:
            for i in range(len(indZeroVariance)):
                indNonZero = np.where(finalCandidates[indZeroVariance[i],:]!=0)[0]
                if len(indNonZero)>1:
                    continue
                else:
                    for j in range(len(indNonZero)):
                        for k in range(len(finalCandidates)):
                            if k!=indZeroVariance[i]:
                                finalCandidates[k,indNonZero[j]] = 0
        for i in range(len(finalCandidates)):
            finalCandidates_std[i] = np.nanstd(np.sum(finalCandidates[i,:]*v_counts_subtree,axis=1))
            finalCandidates_outside_std[i] = np.nanstd(np.sum(finalCandidates[i,:]*v_counts_outside,axis=1))

        # change the sign of the constraint vectors so that the first non-zero element is positive
        for i in range(len(finalCandidates)):
            indNonZero = np.where(finalCandidates[i,:]!=0)[0][0]
            if finalCandidates[i,indNonZero]<0:
                finalCandidates[i,:] = -finalCandidates[i,:]
                
        # make sure we reduce/simplify the constraint vectors as much as possible
        for i in range(len(finalCandidates)):
            finalCandidates[i,:] = finalCandidates[i,:]/np.min(np.abs(finalCandidates[i,finalCandidates[i,:]!=0]))
            
        # get the final values of the variation for these constraints outside the subtree
        finalCandidates_outside_std = []
        for i in range(len(finalCandidates)):
            finalCandidates_outside_std.append(np.nanstd(np.sum(finalCandidates[i,:]*v_counts_outside,axis=1)))
        finalCandidates_outside_std = np.array(finalCandidates_outside_std)
        
        # if any constraints now for some reason don't meet the threshold inside or outside, remove!
        indRemove = np.where((finalCandidates_std**2/totalVar>constraintThresholdLocal) & ((finalCandidates_std**2)/(finalCandidates_outside_std**2)>constraintRatioHumanReadable))[0]
        if len(indRemove)>0:
            finalCandidates = np.delete(finalCandidates,indRemove,axis=0)
            finalCandidates_std = np.delete(finalCandidates_std,indRemove,axis=0)
            finalCandidates_outside_std = np.delete(finalCandidates_outside_std,indRemove,axis=0)
        
        # sort one more time by the ratio of the constraint var to the total var
        indSort = np.argsort(finalCandidates_std**2/totalVar)
        finalCandidates = finalCandidates[indSort,:]
        finalCandidates_std = finalCandidates_std[indSort]
        finalCandidates_outside_std = np.array(finalCandidates_outside_std)[indSort]
        finalCandidates_norm = np.nan*finalCandidates_std
        finalCandidates_outside_norm = np.nan*finalCandidates_std
            
            
        #--------------------------------- plasticity ----------------------------------------------

        # now the same (similar) for the plasticity
        
        # reduce to those which satisfy two constraints: the variance ratio and the outside variance ratio
        indReduce = []
        for i in range(len(cvec_unique)):
            if (in_value_constrained_std[i]**2>totalVar*plasticityThresholdLocal) | (in_value_constrained_std[i]**2>(constrained_outside_std[i]**2)*plasticityRatioHumanReadable):
                indReduce.append(i)
        candidates_plasticity = cvec_unique[indReduce,:]
        candidates_plasticity_std = in_value_constrained_std[indReduce]
        candidates_plasticity_outside_std = np.array(constrained_outside_std)[indReduce]

        # first sort
        candidates_plasticity_sort_ind = np.flipud(np.argsort(candidates_plasticity_std**2/totalVar))
        candidates_plasticity_std = candidates_plasticity_std[candidates_plasticity_sort_ind]
        candidates_plasticity_outside_std = candidates_plasticity_outside_std[candidates_plasticity_sort_ind]
        candidates_plasticity = candidates_plasticity[candidates_plasticity_sort_ind,:]
        # now reduce to the linearly independent vectors, up through the number of non-readable constraints found
        _, inds = sp.Matrix(candidates_plasticity).T.rref()   # to check the rows you need to transpose!
        finalCandidatesPlasticity = candidates_plasticity[inds,:]
        finalCandidatesPlasticity_std = np.ndarray.flatten(candidates_plasticity_std[inds,None])
        finalCandidatesPlasticity_outside_std = np.ndarray.flatten(candidates_plasticity_outside_std[inds,None])

        # remove any single coefficient constraints found above in finalCandidates from the plasticity vectors
        # they are just dead weights!
        for i in range(len(finalCandidates)):
            indNonZero = np.where(finalCandidates[i,:]!=0)[0]
            # if number of nonzero coefficients is equal to one
            if len(np.where(finalCandidates[i,:]!=0)[0])==1:
                # remove the nonzero coefficient from finalCandidatesPlasticity
                finalCandidatesPlasticity[:,indNonZero] = 0

        # recalculate the std and outside std
        finalCandidatesPlasticity_std = np.sqrt(np.var(np.dot(finalCandidatesPlasticity,v_counts_subtree.T),axis=1))/np.sqrt(np.sum(finalCandidatesPlasticity**2,axis=1))
        finalCandidatesPlasticity_outside_std = np.sqrt(np.var(np.dot(finalCandidatesPlasticity,v_counts_outside.T),axis=1))/np.sqrt(np.sum(finalCandidatesPlasticity**2,axis=1))

        # change the sign of the constraint vectors so that the first non-zero element is positive
        # if all coefficients are nonzero then skip this! and just set the first coefficient to be positive
        for i in range(len(inds)):
            # check if all coefficients are nonzero or not
            if len(np.where(finalCandidatesPlasticity[i,:]!=0)[0])>1:
                indNonZero = np.where(finalCandidatesPlasticity[i,:]!=0)[0][0]
                if finalCandidatesPlasticity[i,indNonZero]<0:
                    finalCandidatesPlasticity[i,:] = -finalCandidatesPlasticity[i,:]
            else:
                if finalCandidatesPlasticity[i,0]<0:
                    finalCandidatesPlasticity[i,:] = -finalCandidatesPlasticity[i,:]
                
        # keep only unique rows (after removing constraints and simplifying some may be the same)
        finalCandidatesPlasticity, inds = np.unique(finalCandidatesPlasticity,axis=0,return_index=True)
        finalCandidatesPlasticity_std = finalCandidatesPlasticity_std[inds]
        finalCandidatesPlasticity_outside_std = finalCandidatesPlasticity_outside_std[inds]
            
        # get the final values of the variation for these constraints outside the subtree
        finalCandidatesPlasticity_outside_std = []
        for i in range(len(finalCandidatesPlasticity)):
            finalCandidatesPlasticity_outside_std.append(np.nanstd(np.sum(finalCandidatesPlasticity[i,:]*v_counts_outside,axis=1)))
        finalCandidatesPlasticity_outside_std = np.array(finalCandidatesPlasticity_outside_std)
        
        # sort one more time by the ratio of the constraint std to the outside std
        indSort = np.argsort(finalCandidatesPlasticity_std**2/totalVar)
        finalCandidatesPlasticity = finalCandidatesPlasticity[indSort,:]
        finalCandidatesPlasticity_std = finalCandidatesPlasticity_std[indSort]
        finalCandidatesPlasticity_outside_std = np.array(finalCandidatesPlasticity_outside_std)[indSort]
        
    else: # the full tree
        
        # multiply cvec_unique, which is, for example, (8403,5), by v_counts_subtree, which is (388,5),
        # to get a new matrix that is (8403,388)
        
        # get norm instead of std
        # the norm is the addition of all the individual variances for each coefficient of the potential constraint
        # multiplied by the vertebral counts of that coefficient's column
        # then take the square root of that sum
        in_value_constrained_norm = []
        for i in range(len(cvec_unique)):
            in_value_constrained_norm.append(np.sqrt(np.sum(np.nanvar(cvec_unique[i,:]*v_counts_subtree,axis=0))))
        in_value_constrained_norm = np.array(in_value_constrained_norm)
        
        # reduce
        candidates = cvec_unique[in_value_constrained_std<in_value_constrained_norm,:]
        candidates_std = in_value_constrained_std[in_value_constrained_std<in_value_constrained_norm]
        candidates_norm = in_value_constrained_norm[in_value_constrained_std<in_value_constrained_norm]
    
        # reduce to the linearly independent vectors, sorting by lowest candidates_std divided by the norm
        # first sort
        # now sort not just by the std ratio but also by the "complexity" of the constraint (size of largest coefficient)
        complexity = np.max(np.abs(candidates),axis=1)
        candidates_std_sort_ind = np.lexsort((candidates_std/candidates_norm,complexity))
        candidates_std = candidates_std[candidates_std_sort_ind]
        candidates_norm = candidates_norm[candidates_std_sort_ind]
        candidates = candidates[candidates_std_sort_ind,:]
        # now reduce to the linearly independent vectors, up through the number of non-readable constraints found
        _, inds = sp.Matrix(candidates).T.rref()  # to check the rows you need to transpose!

        # define final candidates
        finalCandidates = candidates[inds,:]
        finalCandidates_std = np.ndarray.flatten(candidates_std[inds,None])
        finalCandidates_norm = np.ndarray.flatten(candidates_norm[inds,None])
        finalCandidates_outside_std = 0*finalCandidates_std
        
        finalCandidatesPlasticity = []
        finalCandidatesPlasticity_std = 0
        finalCandidatesPlasticity_outside_std = 0
        finalCandidates_norm = 0
        finalCandidates_outside_norm = 0
    
    # save
    if np.size(finalCandidates)>0:
        constraint_df = pd.DataFrame(finalCandidates,columns=toCheck)
        constraint_df['std'] = finalCandidates_std
        constraint_df['outside_std'] = finalCandidates_outside_std
        constraint_df['totalVar'] = [totalVar]*len(finalCandidates)
        constraint_df['constraintRatio'] = constraintRatioHumanReadable
        constraint_df['constraintThresholdLocal'] = constraintThresholdLocal
        constraint_df['span'] = spanHumanReadable
        constraint_df.to_csv(datadir+'constraints_humanReadable.csv',index=False)
    
    # save
    if np.size(finalCandidatesPlasticity)>0:
        plasticity_df = pd.DataFrame(finalCandidatesPlasticity,columns=toCheck)
        plasticity_df['std'] = finalCandidatesPlasticity_std
        plasticity_df['outside_std'] = finalCandidatesPlasticity_outside_std
        plasticity_df['totalVar'] = [totalVar]*len(finalCandidatesPlasticity)
        plasticity_df['plasticityRatio'] = plasticityRatioHumanReadable
        plasticity_df['constraintThresholdLocal'] = constraintThresholdLocal
        plasticity_df['span'] = spanHumanReadable
        plasticity_df.to_csv(datadir+'plasticity_humanReadable.csv',index=False)
    
    return finalCandidates,finalCandidates_std,finalCandidates_norm,finalCandidates_outside_std,finalCandidates_outside_norm,finalCandidatesPlasticity,finalCandidatesPlasticity_std,finalCandidatesPlasticity_outside_std

#%% function for plotting the human-readable constraint candidates

def plotReadableConstraints(candidates,candidates_std,candidates_norm,candidates_outside_std,candidates_outside_norm,v_counts_subtree,species,full_tree,globalSpecies,featureNames,class_subtree,class_global,datadir,toCheck):

    # we want to also do something for the full tree (so when len(globalSpecies) = 0)
    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    v_counts_outside = full_tree.loc[:,pd.Index(toCheck)].values
    if len(indRemove)<len(globalSpecies):
        v_counts_outside = np.delete(v_counts_outside, indRemove, axis=0)
        class_global = np.delete(class_global, indRemove, axis=0)

    totalVar = np.trace(np.cov(v_counts_subtree.T))
    
    # make a nice constraint name
    if featureNames[0]=='Cervical':
        constraintNames = ['C','T','L','S','Ca']
    else:
        constraintNames = featureNames
        
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
    
    if featureNames[0]=='Cervical':
        markerWheel = ['o','s','^','>'] # one for each class
        
    classWheel = ['Mammalia','Aves','Reptilia','Amphibia']
        
    for i in range(len(candidates)):
            
        constraintText = ''
        for j in range(len(candidates[i])):
            if candidates[i][j]!=0:
                if candidates[i][j]>0:
                    constraintText = constraintText + ' + '
                else:
                    constraintText = constraintText + ' - '
                constraintText = constraintText + str(np.abs(candidates[i][j])) + constraintNames[j]
        constraintText = constraintText[3:]

        fig, ax = plt.subplots(figsize=(5,5))

        ind_outside = np.linspace(0,len(v_counts_outside)-1,len(v_counts_subtree)).astype(int)
        values_subtree = np.zeros(len(v_counts_subtree))
        values_outside = np.zeros(len(v_counts_subtree))
        values_global = np.zeros(len(v_counts_outside))
        indNonZero = np.where(candidates[i,:]!=0)[0]
        if len(species) != len(globalSpecies):
            for j in range(np.size(indNonZero)):
                values_subtree = values_subtree + candidates[i,indNonZero[j]]*v_counts_subtree[:,indNonZero[j]]
                values_outside = values_outside + candidates[i,indNonZero[j]]*v_counts_outside[ind_outside,indNonZero[j]]
                values_global = values_global + candidates[i,indNonZero[j]]*v_counts_outside[:,indNonZero[j]]
            for j in range(len(uniqueClasses)):
                indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
                plt.plot(values_outside[indClass],values_subtree[indClass],color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],marker=markerWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],linestyle='None',alpha=0.5,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
            plt.ylim(np.min([0,np.min(values_outside)-1]),np.max([np.max(values_subtree)+1,np.max(values_outside)+1]))
            plt.xlabel(f'values outside subtree for constraint: {constraintText}')
            plt.ylabel(f'values inside subtree for constraint: {constraintText}')
        else: # full tree
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
                values0 = values0 + candidates[i,group0[j]]*v_counts_subtree[:,group0[j]]
                if candidates[i,group0[j]]>0:
                    group0Name = group0Name + ' + '
                else:
                    group0Name = group0Name + ' - '
                group0Name = group0Name + str(np.abs(candidates[i,group0[j]])) + constraintNames[group0[j]]
            group0Name = group0Name[3:]
            values1 = np.zeros(len(v_counts_subtree))
            group1Name = ''
            if candidates[i,group1[0]]>0: # to make sure the first coefficient is positive and we aren't plotting negative values against positive ones
                prefactor = 1
            else:
                prefactor = -1
            for j in range(len(group1)):
                values1 = values1 + prefactor*candidates[i,group1[j]]*v_counts_subtree[:,group1[j]]
                if candidates[i,group1[j]]>0:
                    group1Name = group1Name + ' + '
                else:
                    group1Name = group1Name + ' - '
                group1Name = group1Name + str(np.abs(candidates[i,group1[j]])) + constraintNames[group1[j]]
            group1Name = group1Name[3:]
            for j in range(len(uniqueClasses)):
                indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
                plt.plot(values0[indClass],values1[indClass],color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],marker=markerWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],linestyle='None',alpha=0.5,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
            plt.xlabel(f'{group0Name}')
            plt.ylabel(f'{group1Name}')
        
        plt.legend(frameon=False)

        plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma^2/\\sigma_{total}^2}$ = " + f" {str(np.round(candidates_std[i]**2/totalVar,2))}\nfor {textClass}")
        plt.savefig(datadir+'constraint_humanReadable_'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()
        
        # also plot the distributions of the constraint values if not looking at the full tree
        if len(species) != len(globalSpecies):
            
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
            plt.title(f"constraint: {constraintText}, " + "$\mathregular{\\sigma^2/\\sigma_{total}^2}$ = " + f" {str(np.round(candidates_std[i]**2/totalVar,2))}\nfor {textClass}")
            plt.savefig(datadir+'constraint_humanReadable_hist_'+str(i)+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
        
#%% function for plotting the human-readable constraint candidates

def plotReadablePlasticity(candidates,candidates_std,candidates_outside_std,v_counts_subtree,species,full_tree,globalSpecies,featureNames,class_subtree,datadir,toCheck):

    indRemove = [i for i, s in enumerate(globalSpecies) if s in species]
    v_counts_outside = full_tree.loc[:,pd.Index(toCheck)].values
    v_counts_outside = np.delete(v_counts_outside, indRemove, axis=0)

    totalVar = np.trace(np.cov(v_counts_subtree.T))
    
    # make a nice constraint name
    if featureNames[0]=='Cervical':
        constraintNames = ['C','T','L','S','Ca']
    else:
        constraintNames = featureNames
        
    # get the number in the each unique class
    uniqueClasses = np.unique(class_subtree)
    numClasses = np.zeros(len(uniqueClasses))
    for i in range(len(uniqueClasses)):
        numClasses[i] = np.sum(np.array(class_subtree)==uniqueClasses[i])
    textClass = ''
    for j in range(len(uniqueClasses)):
        textClass = textClass + uniqueClasses[j] + ' (' + str(int(numClasses[j])) + ')' + ', '
    textClass = textClass[:-2]
    
    if featureNames[0]=='Cervical':
        markerWheel = ['o','s','^','>'] # one for each class
        
    classWheel = ['Mammalia','Aves','Reptilia','Amphibia']
        
    for i in range(len(candidates)):
            
        constraintText = ''
        for j in range(len(candidates[i])):
            if candidates[i][j]!=0:
                if candidates[i][j]>0:
                    constraintText = constraintText + ' + '
                else:
                    constraintText = constraintText + ' - '
                constraintText = constraintText + str(np.abs(candidates[i][j])) + constraintNames[j]
        constraintText = constraintText[3:]

        fig, ax = plt.subplots(figsize=(5,5))

        ind_outside = np.linspace(0,len(v_counts_outside)-1,len(v_counts_subtree)).astype(int)
        values_subtree = np.zeros(len(v_counts_subtree))
        values_outside = np.zeros(len(v_counts_subtree))
        indNonZero = np.where(candidates[i,:]!=0)[0]
        for j in range(len(indNonZero)):
            values_subtree = values_subtree + candidates[i,indNonZero[j]]*v_counts_subtree[:,indNonZero[j]]
            values_outside = values_outside + candidates[i,indNonZero[j]]*v_counts_outside[ind_outside,indNonZero[j]]
        for j in range(len(uniqueClasses)):
            indClass = np.where(np.array(class_subtree)==uniqueClasses[j])[0]
            plt.plot(values_outside[indClass],values_subtree[indClass],color=colorWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],marker=markerWheel[np.where(np.array(classWheel)==uniqueClasses[j])[0][0]],linestyle='None',alpha=0.5,label=uniqueClasses[j]+' ('+str(int(numClasses[j]))+')')
        plt.ylim(np.min([0,np.min(values_outside)-1]),np.max([np.max(values_subtree)+1,np.max(values_outside)+1]))
        plt.xlabel(f'values outside subtree for plasticity: {constraintText}')
        plt.ylabel(f'values inside subtree for plasticity: {constraintText}')
        
        plt.legend(frameon=False)

        plt.title(f"plasticity: {constraintText}, " + "$\mathregular{\\sigma^2/\\sigma_{total}^2}$ = " + f" {str(np.round(candidates_std[i]**2/totalVar,2))}\nfor {textClass}")
        plt.savefig(datadir+'plasticity_humanReadable_'+str(i)+'.png',dpi=300,bbox_inches='tight')
        plt.close()
