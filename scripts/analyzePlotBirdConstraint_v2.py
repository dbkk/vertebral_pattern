# Analyze and plot the bird and ancient theropod data
# Fig. 3 in the paper
# and Extended Data Fig. 3
# v2

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath
outputPath = inputPath

#%% import libraries

import numpy as np
import pandas as pd
import glob
import scipy.stats
from matplotlib import gridspec
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
markerWheel = ['o','s','^','>','<','v','d'] # pour convenience
fontSize = 14
markerSize = 6
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
colorMap = 'coolwarm' # or 'gray' or 'bwr' or 'RdBu'

#%% load the data

# load the full vertebral formula data
dataPath = glob.glob(inputPath+'vertebralFormulaOrdered_v2.csv')[0]
vertebral = pd.read_csv(dataPath)
mammals = vertebral[vertebral['Class']=='Mammalia']
birds = vertebral[vertebral['Class']=='Aves']
reptiles = vertebral[vertebral['Class']=='Reptilia']
amphibians = vertebral[vertebral['Class']=='Amphibia']
# load all the eyton data
eyton = pd.read_csv(inputPath+'additionalData/eytonBirds.csv')
# load all the abiko data
abiko = pd.read_csv(inputPath+'additionalData/abikoBirdMuseum.csv')
# load the ancient bird (theropod) data
ancientBirds = pd.read_csv(inputPath+'additionalData/ancientBirdData.csv')
# load the bats
bats = pd.read_csv(inputPath+'additionalData/batData.csv')
# load the pterosaurs
pterosaurs = pd.read_csv(inputPath+'additionalData/pterosaurData.csv')

# plot the theropod and bird data according to the tree from Brusatte et al. 2014

#%% individual vertebrae

fontSizeHere = 28

colorWheel_CB = [colorWheel[0],colorWheel[1],'purple',colorWheel[2],colorWheel[3],colorWheel[3],colorWheel[3],colorWheel[3],colorWheel[3]] 

fig = plt.figure(figsize=(1.5,8))
ax = plt.gca()
alphaValue = 0.7
plt.plot(ancientBirds['Cervical'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[0],alpha=alphaValue,label='C',linewidth=3)
plt.plot(ancientBirds['Thoracic'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[1],alpha=alphaValue,label='T',linewidth=3)
plt.plot(ancientBirds['Lumbar'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[2],alpha=alphaValue,label='L',linewidth=3)
plt.plot(ancientBirds['Sacral'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[3],alpha=alphaValue,label='S',linewidth=3)
plt.plot(ancientBirds['Caudal'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[6],alpha=alphaValue,label='Ca',linewidth=3)
plt.plot(np.zeros(len(ancientBirds)),np.flipud(np.arange(len(ancientBirds))),'k:')
plt.legend(frameon=False,loc=[0.24,0.26],handlelength=0.3,fontsize=fontSizeHere,markerfirst=False,handletextpad=0.15,labelspacing=0.3)
plt.ylim(0,len(ancientBirds)-1)
plt.xlim(-1,50)
plt.xticks((1,25,50))
ax.set_xticklabels((1,25,50),fontsize=fontSizeHere)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks([])

axT = ax.twiny()
# axT.tick_params(direction = 'in')
axT.set_xticks((0,25,50))
axT.set_xticklabels((0,25,50),fontsize=fontSizeHere)
axT.set_xlim(-1,50)

# save
plt.savefig(outputPath+'plots/birdsTheropods_Fig3i_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/birdsTheropods_Fig3i_v2.pdf',dpi=300,bbox_inches='tight')


#%% distal constraints: C+T-S-Ca and C-S

fig = plt.figure(figsize=(2,8))
ax = plt.gca()
alphaValue = 0.7
plt.plot(ancientBirds['Cervical']-ancientBirds['Sacral'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[0],alpha=1.0,label='C-S',linewidth=3,linestyle='--')
plt.plot(ancientBirds['Cervical']+ancientBirds['Thoracic']-ancientBirds['Sacral']-ancientBirds['Caudal'],np.flipud(np.arange(len(ancientBirds))),color=colorWheel_CB[1],alpha=1.0,label='C+T\n-S-Ca',linewidth=3)
plt.plot(np.zeros(len(ancientBirds)),np.flipud(np.arange(len(ancientBirds))),'k:',linewidth=1.5)
# plt.legend(frameon=False,loc='center right',handlelength=0.9,fontsize=fontSize+8)
plt.ylim(0,len(ancientBirds)-1)
plt.xlim(-30,30)
plt.xlabel('C-S, C+T-S-Ca',fontsize=fontSize)
plt.xticks((-20,0,20))
ax.set_xticklabels((-20,0,20),fontsize=fontSize)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks([])

# plot the bird standard deviation for the C+T-S-Ca constraint as a vertical band: varCTSCa
# use the patches rectangle function to plot the variance as a vertical band

# get variation of the C-S and C+T-S-Ca constraints from the allBirdData
birds['C-S'] = birds['Cervical'].values - birds['Sacral'].values
varCS = np.nanstd(birds['C-S'].values)
birds['C+T-S-Ca'] = birds['Cervical'].values + birds['Thoracic'].values - birds['Sacral'].values - birds['Caudal'].values
varCTSCa = np.nanstd(birds['C+T-S-Ca'].values)
meanCTSCa = np.nanmean(birds['C+T-S-Ca'].values)

from matplotlib.patches import Rectangle
# make the rectangle
currentAxis = plt.gca()
# make the rectangle
rect = Rectangle((meanCTSCa-3*varCTSCa,0),3*2*varCTSCa,len(ancientBirds),linewidth=1,edgecolor='None',facecolor='grey',alpha=0.3)
# add the rectangle to the plot
currentAxis.add_patch(rect)
# add text to explain
# plt.text(-25,10,'C+T-S-Ca',fontsize=fontSize+8)

axT = ax.twiny()
# axT.tick_params(direction = 'in')
axT.set_xticks((-20,0,20))
axT.set_xticklabels((-20,0,20),fontsize=fontSize)
axT.set_xlim(-30,30)

# save
plt.savefig(outputPath+'plots/birdsTheropods_Fig3j_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/birdsTheropods_Fig3j_v2.pdf',dpi=300,bbox_inches='tight')

#%% now make the histogram plot of the C+T-S-Ca constraint


# make the figure
fontSize = 14
faceColor = 'white'
markerSize = 6
lineWidth = 0.5
fig, ax = plt.subplots(4,1,figsize=(4,4))

numBins = 50
histRange = 40
# mammals
ax[0].hist(vertebral['Cervical'][vertebral['Class']=='Mammalia']+vertebral['Thoracic'][vertebral['Class']=='Mammalia']-vertebral['Lumbar'][vertebral['Class']=='Mammalia']-vertebral['Sacral'][vertebral['Class']=='Mammalia']-vertebral['Caudal'][vertebral['Class']=='Mammalia'],bins=numBins,range=(-50,50),color=colorWheel[0],alpha=0.5,label='Mammalia',edgecolor='black',linewidth=lineWidth)
# and bats
ax[0].hist(bats['Cervical']+bats['Thoracic']-bats['Lumbar']-bats['Sacral']-bats['Caudal'],bins=numBins,range=(-histRange,histRange),color='black',alpha=0.75,label='Chiroptera',edgecolor='black',linewidth=lineWidth)
ax[0].set_xticks([])
ax[0].legend(loc='upper right',fontsize=fontSize-3,frameon=False,handlelength=0.5,handletextpad=0.3,labelspacing=0.3,markerfirst=False)
ax[0].set_yticks((0,5,10))
ax[0].axvline(x=0,linestyle='--',color='black',linewidth=1)
# birds
# include all abiko
abikoNotInBirds = abiko[~abiko['Species'].isin(birds['Species'])]
birds = pd.concat([birds,abikoNotInBirds])
ax[1].hist(birds['Cervical']+birds['Thoracic']-birds['Lumbar']-birds['Sacral']-birds['Caudal'],bins=numBins,range=(-histRange,histRange),color=colorWheel[1],alpha=0.5,label='Aves',edgecolor='black',linewidth=lineWidth)
ax[1].set_xticks([])
ax[1].set_yticks((0,20,40,60))
ax[1].set_ylim([0,70])
# theropoda
# plot the flying/gliding theropods
xValue = ancientBirds['Cervical'].iloc[1:]+ancientBirds['Thoracic'].iloc[1:]-ancientBirds['Lumbar'].iloc[1:]-ancientBirds['Sacral'].iloc[1:]-ancientBirds['Caudal'].iloc[1:]
yValue = np.ones(len(ancientBirds)-1)*30
ax[1].plot(xValue,yValue,marker='s',color='black',linestyle='None',mfc='None',label='Theropoda',markersize=markerSize,alpha=0.5)
# pterosaurs
xValue = pterosaurs['Cervical']+pterosaurs['Thoracic']-pterosaurs['Lumbar']-pterosaurs['Sacral']-pterosaurs['Caudal']
yValue = np.ones(len(pterosaurs))*60
ax[1].plot(xValue,yValue,marker='^',color='black',linestyle='None',mfc='None',label='Pterosauria',markersize=markerSize,alpha=0.5)

# point out Tyrannosaurus and Archaeopteryx
# plot the label for Archaeopteryx from ancient birds
ind = ancientBirds[ancientBirds['Common Name']=='Archaeopteryx'].index[0]
xValue = ancientBirds['Cervical'].values[ind]+ancientBirds['Thoracic'].values[ind]-ancientBirds['Lumbar'].values[ind]-ancientBirds['Sacral'].values[ind]-ancientBirds['Caudal'].values[ind]
yValue = 30
ax[1].text(xValue-2.25,yValue+10,'Archaeopteryx',fontsize=fontSize-4,ha='right')
# make a little arrow pointing to the point
ax[1].arrow(xValue-2,yValue+10,+2.5,-10,head_width=0,head_length=0,fc='black',ec='black')
# plot the label for Tyrannosaurus from ancient birds which is at index 8, plot this to the left of the point
ind = ancientBirds[ancientBirds['Common Name']=='Tyrannosaurus'].index[0]
xValue = ancientBirds['Cervical'].values[ind]+ancientBirds['Thoracic'].values[ind]-ancientBirds['Lumbar'].values[ind]-ancientBirds['Sacral'].values[ind]-ancientBirds['Caudal'].values[ind]
print(xValue)
yValue = 30
ax[1].text(xValue+7,yValue-25,'Tyrannosaurus',fontsize=fontSize-4,ha='right')
# make a little arrow pointing to the point
ax[1].arrow(xValue-5,yValue-14,+5,14,head_width=0,head_length=0,fc='black',ec='black')

ax[1].legend(loc=([0.64,0.030]),fontsize=fontSize-3,frameon=False,handlelength=0.5,handletextpad=0.3,labelspacing=0.3,markerfirst=False)
ax[1].axvline(x=0,linestyle='--',color='black',linewidth=1)
# plot the label for Archaeopteryx from ancient birds
ind = 0
# reptiles
ax[2].hist(vertebral['Cervical'][vertebral['Class']=='Reptilia']+vertebral['Thoracic'][vertebral['Class']=='Reptilia']-vertebral['Lumbar'][vertebral['Class']=='Reptilia']-vertebral['Sacral'][vertebral['Class']=='Reptilia']-vertebral['Caudal'][vertebral['Class']=='Reptilia'],bins=numBins,range=(-histRange,histRange),color=colorWheel[2],alpha=0.5,label='Reptilia',edgecolor='black',linewidth=lineWidth)
ax[2].set_xticks([])
ax[2].legend(loc='upper right',fontsize=fontSize-3,frameon=False,handlelength=0.5,handletextpad=0.3,labelspacing=0.3,markerfirst=False)
ax[2].axvline(x=0,linestyle='--',color='black',linewidth=1)
# amphibians
ax[3].hist(vertebral['Cervical'][vertebral['Class']=='Amphibia']+vertebral['Thoracic'][vertebral['Class']=='Amphibia']-vertebral['Lumbar'][vertebral['Class']=='Amphibia']-vertebral['Sacral'][vertebral['Class']=='Amphibia']-vertebral['Caudal'][vertebral['Class']=='Amphibia'],bins=numBins,range=(-histRange,histRange),color=colorWheel[3],alpha=0.5,label='Amphibia',edgecolor='black',linewidth=lineWidth)
ax[3].legend(loc='upper right',fontsize=fontSize-3,frameon=False,handlelength=0.5,handletextpad=0.3,labelspacing=0.3,markerfirst=False)
ax[3].axvline(x=0,linestyle='--',color='black',linewidth=1)
# ax[3].set_yticks((0,20,40))
ax[3].set_ylim([0,8])

ax[3].set_xlabel('(Cervical+Thoracic)-(Lumbar+Sacral+Caudal)',fontsize=fontSize,x=0.40)
plt.text(-0.18,1.05,'Frequency',fontsize=fontSize,ha='center',va='center',rotation=90,transform=ax[2].transAxes)

# save
plt.savefig(outputPath+'plots/birdConstraintHistogram_Fig3k_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/birdConstraintHistogram_Fig3k_v2.pdf',dpi=300,bbox_inches='tight')


#%% reduce Eyton and Abiko to only those with the full formula
# also get those entries not in either

# reduce eyton to those with non-NaN values for Cervical, Thoracic, Sacral, and Caudal
eyton = eyton[(eyton['Cervical'].notna())&(eyton['Thoracic'].notna())&(eyton['Sacral'].notna())&(eyton['Caudal'].notna())]
# reset the index
eyton = eyton.reset_index(drop=True)

# do the same for abiko
abiko = abiko[(abiko['Cervical'].notna())&(abiko['Thoracic'].notna())&(abiko['Sacral'].notna())&(abiko['Caudal'].notna())]
# reset the index
abiko = abiko.reset_index(drop=True)

# reduce from those in eyton or abiko by comparing species
speciesEyton = eyton['Species'].unique()
speciesAbiko = abiko['Species'].unique()
birds = vertebral[vertebral['Class']=='Aves']
birdsNotInEytonOrAbiko = birds[~birds['Species'].isin(speciesEyton)]
birdsNotInEytonOrAbiko = birdsNotInEytonOrAbiko[~birdsNotInEytonOrAbiko['Species'].isin(speciesAbiko)]
# reset the index
birdsNotInEytonOrAbiko = birdsNotInEytonOrAbiko.reset_index(drop=True)

#%% get the combined species in eyton and abiko

#%% get the counts for these two datasets

# sort abiko by cervical
speciesShared = abiko['Species'].unique()

df = np.zeros((len(speciesShared),5))
indShared = np.zeros((len(speciesShared)))
for i in range(len(speciesShared)):
    if len(eyton[eyton['Species']==speciesShared[i]])==0:
        continue
    else:
        indShared[i] = 1
        df[i,0] = eyton[eyton['Species']==speciesShared[i]]['Cervical'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Cervical'].values[0]
        df[i,1] = eyton[eyton['Species']==speciesShared[i]]['Thoracic'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Thoracic'].values[0]
        df[i,2] = eyton[eyton['Species']==speciesShared[i]]['Sacral'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Sacral'].values[0]
        df[i,3] = eyton[eyton['Species']==speciesShared[i]]['Caudal'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Caudal'].values[0]
        # difference with reference to bird constraint: C+T-S-Ca = 0
        df[i,4] = (eyton[eyton['Species']==speciesShared[i]]['Cervical'].values[0]+eyton[eyton['Species']==speciesShared[i]]['Thoracic'].values[0]-eyton[eyton['Species']==speciesShared[i]]['Sacral'].values[0]-eyton[eyton['Species']==speciesShared[i]]['Caudal'].values[0])-(abiko[abiko['Species']==speciesShared[i]]['Cervical'].values[0]+abiko[abiko['Species']==speciesShared[i]]['Thoracic'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Sacral'].values[0]-abiko[abiko['Species']==speciesShared[i]]['Caudal'].values[0])
        
# reduce to those with indShared = 1
speciesShared = speciesShared[indShared==1]
df = df[indShared==1,:]

#%% combine the data source and different orders of birds plots into two subplots

# make a figure
fig = plt.figure(figsize=(9,4))
gs = gridspec.GridSpec(1, 2)
# increase width between subplots
gs.update(wspace=0.35)

# plot the data for the different sources
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[0,1])

# plot the data for the different sources
# plot x=y line
ax0.plot([12,31],[12,31],'k--',linewidth=1.5)
ax1.plot([12,31],[12,31],'k--',linewidth=1.5)

# # plot eyton
rEyton,pEyton = scipy.stats.pearsonr(eyton['Cervical']+eyton['Thoracic'],eyton['Sacral']+eyton['Caudal'])
ax0.plot(eyton['Cervical']+eyton['Thoracic'],eyton['Sacral']+eyton['Caudal'],marker=markerWheel[0],color=colorWheel[0],linestyle='None',label='Eyton, r='+str(np.round(rEyton,2))+', p='+"{:.2e}".format(pEyton),alpha=0.2)
# plot abiko
rAbiko,pAbiko = scipy.stats.pearsonr(abiko['Cervical']+abiko['Thoracic'],abiko['Sacral']+abiko['Caudal'])
ax0.plot(abiko['Cervical']+abiko['Thoracic'],abiko['Sacral']+abiko['Caudal'],marker=markerWheel[1],color=colorWheel[1],linestyle='None',label='Abiko, r='+str(np.round(rAbiko,2))+', p='+"{:.2e}".format(pAbiko),alpha=0.2)

rOther,pOther = scipy.stats.pearsonr(birdsNotInEytonOrAbiko['Cervical']+birdsNotInEytonOrAbiko['Thoracic'],birdsNotInEytonOrAbiko['Sacral']+birdsNotInEytonOrAbiko['Caudal'])
ax0.plot(birdsNotInEytonOrAbiko['Cervical']+birdsNotInEytonOrAbiko['Thoracic'],birdsNotInEytonOrAbiko['Sacral']+birdsNotInEytonOrAbiko['Caudal'],marker=markerWheel[2],color=colorWheel[2],linestyle='None',label='Other, r='+str(np.round(rOther,2))+', p='+"{:.2e}".format(pOther),alpha=0.2)

# plot the lines connecting the matches between eyton and abiko
for i in range(len(speciesShared)):
    # plot the eyton one, then the abiko one, and then the line connecting them
    if i == -1:
        # eyton
        ax0.plot(eyton[eyton['Species']==speciesShared[i]]['Cervical']+eyton[eyton['Species']==speciesShared[i]]['Thoracic'],eyton[eyton['Species']==speciesShared[i]]['Sacral']+eyton[eyton['Species']==speciesShared[i]]['Caudal'],marker=markerWheel[0],color=colorWheel[0],linestyle='None',label='Eyton',alpha=0.5)
        # abiko
        ax0.plot(abiko[abiko['Species']==speciesShared[i]]['Cervical']+abiko[abiko['Species']==speciesShared[i]]['Thoracic'],abiko[abiko['Species']==speciesShared[i]]['Sacral']+abiko[abiko['Species']==speciesShared[i]]['Caudal'],marker=markerWheel[1],color=colorWheel[1],linestyle='None',label='Abiko',alpha=0.5)
    else:
        # eyton
        ax0.plot(eyton[eyton['Species']==speciesShared[i]]['Cervical']+eyton[eyton['Species']==speciesShared[i]]['Thoracic'],eyton[eyton['Species']==speciesShared[i]]['Sacral']+eyton[eyton['Species']==speciesShared[i]]['Caudal'],marker=markerWheel[0],color=colorWheel[0],linestyle='None',alpha=0.5)
        # abiko
        ax0.plot(abiko[abiko['Species']==speciesShared[i]]['Cervical']+abiko[abiko['Species']==speciesShared[i]]['Thoracic'],abiko[abiko['Species']==speciesShared[i]]['Sacral']+abiko[abiko['Species']==speciesShared[i]]['Caudal'],marker=markerWheel[1],color=colorWheel[1],linestyle='None',alpha=0.5)
    # line connecting them
    ax0.plot([abiko[abiko['Species']==speciesShared[i]]['Cervical'].values[0]+abiko[abiko['Species']==speciesShared[i]]['Thoracic'].values[0],eyton[eyton['Species']==speciesShared[i]]['Cervical'].values[0]+eyton[eyton['Species']==speciesShared[i]]['Thoracic'].values[0]],[abiko[abiko['Species']==speciesShared[i]]['Sacral'].values[0]+abiko[abiko['Species']==speciesShared[i]]['Caudal'].values[0],eyton[eyton['Species']==speciesShared[i]]['Sacral'].values[0]+eyton[eyton['Species']==speciesShared[i]]['Caudal'].values[0]],color='black',linewidth=0.5)
    

ax0.set_xlabel('Cervical+Thoracic',fontsize=fontSize)
ax0.set_ylabel('Sacral+Caudal',fontsize=fontSize)
ax0.set_yticks([15,20,25,30])
    
# legend
leg = ax0.legend(frameon=False,loc='lower right',fontsize=fontSize-5,ncol=1,markerfirst=False,handletextpad=0.1)
for lh in leg.legend_handles: 
    lh.set_alpha(1)
    
# order plot

# combine all the data
allData = pd.concat([eyton,abiko,birdsNotInEytonOrAbiko])
# sort by Cervical
allData = allData.sort_values(by=['Cervical'],ascending=False)
# reset the index
allData = allData.reset_index(drop=True)

orderUnique = allData['Order'].unique()

# make a grey scale range for the different orders and there are len(orderUnique)
# make a list of colors
colorWheelGray = []
for i in range(len(orderUnique)):
    colorWheelGray.append((i+1)/(1.15*len(orderUnique)))

# r-value for each order
rOrder = np.zeros((len(orderUnique)))
pOrder = np.zeros((len(orderUnique)))
meanDeviation = np.zeros((len(orderUnique)))
meanDeviationMean = np.sqrt(np.nanmean((allData['Cervical']+allData['Thoracic']-allData['Sacral']-allData['Caudal'])**2))
for i in range(len(orderUnique)):
    # plot the data for this order
    # length longer than 2
    if len(allData['Cervical'][allData['Order']==orderUnique[i]])>2:
        r,p = scipy.stats.pearsonr(allData['Cervical'][allData['Order']==orderUnique[i]]+allData['Thoracic'][allData['Order']==orderUnique[i]],allData['Sacral'][allData['Order']==orderUnique[i]]+allData['Caudal'][allData['Order']==orderUnique[i]])
        rOrder[i] = r
        pOrder[i] = p
    else:
        rOrder[i] = np.nan
        pOrder[i] = np.nan
    meanDeviation[i] = np.sqrt(np.nanmean((allData['Cervical'][allData['Order']==orderUnique[i]]+allData['Thoracic'][allData['Order']==orderUnique[i]]-allData['Sacral'][allData['Order']==orderUnique[i]]-allData['Caudal'][allData['Order']==orderUnique[i]])**2))
    ax1.plot(allData['Cervical'][allData['Order']==orderUnique[i]]+allData['Thoracic'][allData['Order']==orderUnique[i]],allData['Sacral'][allData['Order']==orderUnique[i]]+allData['Caudal'][allData['Order']==orderUnique[i]],marker=markerWheel[i%len(markerWheel)],color=str(colorWheelGray[i]),linestyle='None',label=orderUnique[i],markersize=markerSize)#+', $\mathregular{(\\Delta - \\langle \\Delta \\rangle)/\\langle \\Delta \\rangle}$='+str(np.round((meanDeviation[i]-meanDeviationMean)/meanDeviationMean,2)))

# take the average of the Cervical, Thoracic, Lumbar, Sacral, and Caudal over all duplicate species in allData
# and reduce to only unique species (retaining the average values)
# only do the average on those numerical columns
duplicateData = allData[allData.duplicated(subset=['Species'],keep=False)]
meanDeviationDuplicateData = np.sqrt(np.nanmean((duplicateData['Cervical']+duplicateData['Thoracic']-duplicateData['Sacral']-duplicateData['Caudal'])**2))
# now take the average
duplicateSpeciesList = duplicateData['Species'].unique()
C = []
T = []
S = []
Ca = []
for i in range(len(duplicateSpeciesList)):
    C.append(np.nanmean(duplicateData['Cervical'][duplicateData['Species']==duplicateSpeciesList[i]]))
    T.append(np.nanmean(duplicateData['Thoracic'][duplicateData['Species']==duplicateSpeciesList[i]]))
    S.append(np.nanmean(duplicateData['Sacral'][duplicateData['Species']==duplicateSpeciesList[i]]))
    Ca.append(np.nanmean(duplicateData['Caudal'][duplicateData['Species']==duplicateSpeciesList[i]]))
# make a new dataframe
duplicateDataMean = pd.DataFrame({'Species':duplicateSpeciesList,'Cervical':C,'Thoracic':T,'Sacral':S,'Caudal':Ca})
meanDeviationDuplicateDataMean = np.sqrt(np.nanmean((duplicateDataMean['Cervical']+duplicateDataMean['Thoracic']-duplicateDataMean['Sacral']-duplicateDataMean['Caudal'])**2))
    
print('mean deviation of all data: '+str(meanDeviationMean))
print('mean deviation of duplicate data: '+str(meanDeviationDuplicateData))
print('mean deviation of duplicate data mean: '+str(meanDeviationDuplicateDataMean))
    
ax1.set_xlim([12,45])
ax1.set_ylim([12,32])
ax1.set_yticks([15,20,25,30])
ax1.set_xticks([15,20,25,30,35,40])
# set fontsize
ax1.tick_params(axis='both', which='major', labelsize=fontSize)

ax1.set_xlabel('Cervical+Thoracic',fontsize=fontSize)
ax1.set_ylabel('Sacral+Caudal',fontsize=fontSize)


# plot x=y line
ax1.plot([12,30],[12,30],'k--',linewidth=1.5)

# correlation
r,p = scipy.stats.pearsonr(allData['Cervical']+allData['Thoracic'],allData['Sacral']+allData['Caudal'])

# split the legend
# legend 
handles,labels = ax1.get_legend_handles_labels()
# 0 is the last fourth, 1 is the remaining three fourths
handles0 = handles[3*len(handles)//4:]
labels0 = labels[3*len(labels)//4:]
handles1 = handles[:3*len(handles)//4]
labels1 = labels[:3*len(labels)//4]
legend0 = plt.legend(handles0,labels0,loc=[0.32,0.02],frameon=False, ncol=1, columnspacing=0.4, handlelength=1.5,fontsize=fontSize-7,labelspacing=0.30,markerfirst=False,handletextpad=0.1)
legend1 = plt.legend(handles1,labels1,loc='lower right',frameon=False, ncol=1, columnspacing=0.4, handlelength=1.5,fontsize=fontSize-7,labelspacing=0.30,markerfirst=False,handletextpad=0.1)
ax1.add_artist(legend0)

# plot subplot labels
ax0.text(-0.2,0.98,'A',fontsize=fontSize+4,fontweight='normal',transform=ax0.transAxes)
ax1.text(-0.2,0.98,'B',fontsize=fontSize+4,fontweight='normal',transform=ax1.transAxes)

# save
plt.savefig(outputPath+'plots/birdConstraint_extendedDataFig3_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/birdConstraint_extendedDataFig3_v2.pdf',dpi=300,bbox_inches='tight')