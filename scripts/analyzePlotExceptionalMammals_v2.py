# analyze and plot exceptional mammalian vertebral formulae (sloths, cetaceans, manatees)
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
    "grey",
]
markerWheel = ['o','s','^','>','<','v','d'] # pour convenience
fontSize = 12
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

# load the full vertebral formula data, which is the only .csv file in the inputPath, so search for it with glob
dataPath = glob.glob(inputPath+'vertebralFormulaOrdered_v2.csv')[0]
vertebralData = pd.read_csv(dataPath)
sloth = vertebralData[vertebralData['Species'].str.contains('Bradypus tridactylus')]
mammals = vertebralData[vertebralData['Class']=='Mammalia']
# remove sloths from mammals
mammals = mammals[~mammals['Species'].str.contains('Bradypus tridactylus')]

# load the buchholtz sloth data
buchholtz = pd.read_csv(inputPath+'additionalData/buchholtzSloths.csv')
# add a genus column
buchholtz['genus'] = [x.split(' ')[0] for x in buchholtz['species']]

# load the manatee data from various sources
manatee = pd.read_csv(inputPath+'additionalData/manatee.csv')

# load the buchholtz cetacean data
cetaceans = pd.read_csv(inputPath+'additionalData/buchholtzCetaceans.csv')
# average for each species
cetaceanSpecies = cetaceans['species'].unique()
Clist = []
Tlist = []
Llist = []
Slist = []
Calist = []
for i in range(len(cetaceanSpecies)):
    ind = cetaceans['species'] == cetaceanSpecies[i]
    Clist.append(cetaceans['C'][ind].mean())
    Tlist.append(cetaceans['T'][ind].mean())
    Llist.append(cetaceans['L'][ind].mean())
    Slist.append(cetaceans['S'][ind].mean())
    Calist.append(cetaceans['Ca'][ind].mean())
cetaceansAveragedSpecies = pd.DataFrame({'species':cetaceanSpecies,'C':Clist,'T':Tlist,'L':Llist,'S':Slist,'Ca':Calist})

# load sanchez thoracolumbar data
sanchez = pd.read_csv(inputPath+'additionalData/sanchezThoracolumbar.csv')
afrotheriaOrders = ['Proboscidea','Sirenia','Hyracoidea','Macroscelidea','Afrosoricida','Tubulidentata','Paenungulata','Chrysochloridae','Tenrecidae','Macroscelidia','Sirenia']
# these are all entirely capitalized in sanchez (every letter)
afrotheriaOrdersCapitalized = [x.upper() for x in afrotheriaOrders]
sanchezAfrotheria = sanchez[sanchez['Order'].isin(afrotheriaOrdersCapitalized)]
# remove any with nans in the thoracic or lumbar columns
sanchezAfrotheria = sanchezAfrotheria[~sanchezAfrotheria['Thoracic'].isna()]
sanchezAfrotheria = sanchezAfrotheria[~sanchezAfrotheria['Lumbar'].isna()]


#%% make a plot of all

# two rows and 3 columns gridspec
fig,ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.flatten()
# put some more space between the subplots
plt.subplots_adjust(wspace=0.4,hspace=0.4)

# the first ax[0] will have its axes removed so I can put a sketch in there later
ax[0].axis('off')

# T vs. C
ax[1].plot(mammals['Cervical']-0.1*np.ones(len(mammals)),mammals['Thoracic'],'o',alpha=0.5,color=colorWheel[0],markersize=markerSize,label='Mammalia (C=7)')
# plot Choloepus genus from buchholtz
ax[1].plot(buchholtz['C'][buchholtz['genus']=='Choloepus'],buchholtz['T'][buchholtz['genus']=='Choloepus'],'s',alpha=0.5,color=colorWheel[5],markersize=markerSize,label='Choloepus')
# plot Bradypus genus from buchholtz
ax[1].plot(buchholtz['C'][buchholtz['genus']=='Bradypus'],buchholtz['T'][buchholtz['genus']=='Bradypus'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize,label='Bradypus')
# plot the one Bradypus sloth we have
ax[1].plot(sloth['Cervical'],sloth['Thoracic'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the cetaceans from buchholtz
ax[1].plot(cetaceansAveragedSpecies['C']+0.1*np.ones(len(cetaceansAveragedSpecies)),cetaceansAveragedSpecies['T'],'^',alpha=0.5,color=colorWheel[10],markersize=markerSize+1,label='Cetacea')
# plot the manatees from various sources
ax[1].plot(manatee['C'][manatee['Species']=='Trichechus manatus'],manatee['T'][manatee['Species']=='Trichechus manatus'],'>',color=colorWheel[8],alpha=0.5,markersize=markerSize+1,label='Trichechus')
# plot the afrotherians from Sanchez
ax[1].plot(7*np.ones(len(sanchezAfrotheria)),sanchezAfrotheria['Thoracic'],'<',alpha=0.5,color='yellowgreen',markersize=markerSize+1,label='Afrotheria')
ax[1].set_xlabel('Cervical',fontsize=fontSize)
ax[1].set_ylabel('Thoracic',fontsize=fontSize)
ax[1].set_xlim([4.5,10.5])
# ax[1].set_ylim([8.5,24.5])
ax[1].set_yticks([10,15,20,25])
ax[1].set_ylim([8,29])
ax[1].set_xticks([5,6,7,8,9,10,11,12])
# pearson correlation coefficient for buchholtz all genuses
rSloth,pSloth = scipy.stats.pearsonr(buchholtz['C'],buchholtz['T'])
combinedC = pd.concat([mammals['Cervical'],pd.Series(buchholtz['C'][buchholtz['genus']=='Choloepus'].mean()),pd.Series(buchholtz['C'][buchholtz['genus']=='Bradypus'].mean()),pd.Series(manatee['C'][manatee['Species']=='Trichechus manatus'].mean()),pd.Series(cetaceans['C'].mean())])
combinedT = pd.concat([mammals['Thoracic'],pd.Series(buchholtz['T'][buchholtz['genus']=='Choloepus'].mean()),pd.Series(buchholtz['T'][buchholtz['genus']=='Bradypus'].mean()),pd.Series(manatee['T'][manatee['Species']=='Trichechus manatus'].mean()),pd.Series(cetaceans['T'].mean())])
r,p = scipy.stats.pearsonr(combinedC,combinedT)
ax[1].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+' (inter-Mammalia)\nr='+str(round(rSloth,2))+', p='+"{:.2e}".format(pSloth)+' (intra-Folivora)',fontsize=fontSize)

ax[1].annotate('Mammalia (C=7)',
    xy=(0.94,0.90), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[0],ha='right',fontsize=fontSize-1)
ax[1].annotate('Choloepus',
    xy=(0.94,0.82), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[5],ha='right',fontsize=fontSize-1)
ax[1].annotate('Bradypus',
    xy=(0.94,0.74), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[6],ha='right',fontsize=fontSize-1)
ax[1].annotate('Cetacea',
    xy=(0.94,0.66), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[10],ha='right',fontsize=fontSize-1)
ax[1].annotate('Trichechus',
    xy=(0.94,0.58), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color=colorWheel[8],ha='right',fontsize=fontSize-1)
ax[1].annotate('Afrotheria',
    xy=(0.94,0.50), xycoords='axes fraction',
    xytext=(1.5, 1.5), textcoords='offset points',color='yellowgreen',ha='right',fontsize=fontSize-1)

# L vs. T
ax[2].plot(mammals['Thoracic'],mammals['Lumbar'],'o',alpha=0.5,color=colorWheel[0],markersize=markerSize)
# plot Choloepus genus from buchholtz
ax[2].plot(buchholtz['T'][buchholtz['genus']=='Choloepus'],buchholtz['L'][buchholtz['genus']=='Choloepus'],'s',alpha=0.5,color=colorWheel[5],markersize=markerSize)
# plot Bradypus genus from buchholtz
ax[2].plot(buchholtz['T'][buchholtz['genus']=='Bradypus'],buchholtz['L'][buchholtz['genus']=='Bradypus'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the one Bradypus sloth we have
ax[2].plot(sloth['Thoracic'],sloth['Lumbar'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the cetaceans from buchholtz
ax[2].plot(cetaceansAveragedSpecies['T'],cetaceansAveragedSpecies['L'],'^',alpha=0.5,color=colorWheel[10],markersize=markerSize+1)
# plot the manatees from various sources
ax[2].plot(manatee['T'][manatee['Species']=='Trichechus manatus'],manatee['L'][manatee['Species']=='Trichechus manatus'],'>',color=colorWheel[8],alpha=0.5,markersize=markerSize+1)
# plot the afrotherians from Sanchez
ax[2].plot(sanchezAfrotheria['Thoracic'],sanchezAfrotheria['Lumbar'],'<',alpha=0.5,color='yellowgreen',markersize=markerSize+1,label='Afrotheria')
ax[2].set_xlabel('Thoracic',fontsize=fontSize)
ax[2].set_ylabel('Lumbar',fontsize=fontSize)
ax[2].set_ylim([0,20])
# pearson correlation coefficient for mammals plus sloths plus manatee (all combined)
# for the combination do the average of the bradypus and choloepus and manatee and add them to the mammals
combinedL = pd.concat([mammals['Lumbar'],pd.Series(buchholtz['L'][buchholtz['genus']=='Choloepus'].mean()),pd.Series(buchholtz['L'][buchholtz['genus']=='Bradypus'].mean()),pd.Series(manatee['L'][manatee['Species']=='Trichechus manatus'].mean()),pd.Series(cetaceans['L'].mean())])
r,p = scipy.stats.pearsonr(combinedT,combinedL)
rCetaceans,pCetaceans = scipy.stats.pearsonr(cetaceansAveragedSpecies['T'],cetaceansAveragedSpecies['L'])
rAfrotheria,pAfrotheria = scipy.stats.pearsonr(sanchezAfrotheria['Thoracic'],sanchezAfrotheria['Lumbar'])
ax[2].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+' (inter-Mammalia)\nr='+str(round(rCetaceans,2))+', p='+"{:.2e}".format(pCetaceans)+' (inter-Cetacea)\nr='+str(round(rAfrotheria,2))+', p='+"{:.2e}".format(pAfrotheria)+' (inter-Afrotheria)',fontsize=fontSize)

# S vs. L
ax[3].plot(mammals['Lumbar'],mammals['Sacral'],'o',alpha=0.5,color=colorWheel[0],markersize=markerSize,label='Mammalia (C=7)')
# plot Choloepus genus from buchholtz
ax[3].plot(buchholtz['L'][buchholtz['genus']=='Choloepus'],buchholtz['S'][buchholtz['genus']=='Choloepus'],'s',alpha=0.5,color=colorWheel[5],markersize=markerSize,label='Choloepus')
# plot Bradypus genus from buchholtz
ax[3].plot(buchholtz['L'][buchholtz['genus']=='Bradypus'],buchholtz['S'][buchholtz['genus']=='Bradypus'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize,label='Bradypus')
# plot the one Bradypus sloth we have
ax[3].plot(sloth['Lumbar'],sloth['Sacral'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the cetaceans from buchholtz
ax[3].plot(cetaceansAveragedSpecies['L'],cetaceansAveragedSpecies['S'],'^',alpha=0.5,color=colorWheel[10],markersize=markerSize+1,label='Cetacea')
# plot the manatees from various sources
ax[3].plot(manatee['L'][manatee['Species']=='Trichechus manatus'],manatee['S'][manatee['Species']=='Trichechus manatus'],'>',color=colorWheel[8],alpha=0.5,markersize=markerSize+1,label='Trichechus')
ax[3].set_xlabel('Lumbar',fontsize=fontSize)
ax[3].set_ylabel('Sacral',fontsize=fontSize)
ax[3].set_xlim([0,20])
# combine all (even cetaceans)
combinedS = pd.concat([mammals['Sacral'],pd.Series(buchholtz['S'][buchholtz['genus']=='Choloepus'].mean()),pd.Series(buchholtz['S'][buchholtz['genus']=='Bradypus'].mean()),pd.Series(manatee['S'][manatee['Species']=='Trichechus manatus'].mean()),pd.Series(cetaceans['S'].mean())])
r,p = scipy.stats.pearsonr(combinedL,combinedS)
ax[3].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+' (inter-Mammalia)',fontsize=fontSize)

# Ca vs. S
ax[4].plot(mammals['Sacral'],mammals['Caudal'],'o',alpha=0.5,color=colorWheel[0],markersize=markerSize)
# plot Choloepus genus from buchholtz
ax[4].plot(buchholtz['S'][buchholtz['genus']=='Choloepus'],buchholtz['Cd'][buchholtz['genus']=='Choloepus'],'s',alpha=0.5,color=colorWheel[5],markersize=markerSize)
# plot Bradypus genus from buchholtz
ax[4].plot(buchholtz['S'][buchholtz['genus']=='Bradypus'],buchholtz['Cd'][buchholtz['genus']=='Bradypus'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the one Bradypus sloth we have
ax[4].plot(sloth['Sacral'],sloth['Caudal'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the cetaceans from buchholtz
ax[4].plot(cetaceansAveragedSpecies['S'],cetaceansAveragedSpecies['Ca'],'^',alpha=0.5,color=colorWheel[10],markersize=markerSize+1)
# plot the manatees from various sources
ax[4].plot(manatee['S'][manatee['Species']=='Trichechus manatus'],manatee['Ca'][manatee['Species']=='Trichechus manatus'],'>',color=colorWheel[8],alpha=0.5,markersize=markerSize+1)
ax[4].set_xlabel('Sacral',fontsize=fontSize)
ax[4].set_ylabel('Caudal',fontsize=fontSize)
combinedCa = pd.concat([mammals['Caudal'],pd.Series(buchholtz['Cd'][buchholtz['genus']=='Choloepus'].mean()),pd.Series(buchholtz['Cd'][buchholtz['genus']=='Bradypus'].mean()),pd.Series(manatee['Ca'][manatee['Species']=='Trichechus manatus'].mean()),pd.Series(cetaceans['Ca'].mean())])
r,p = scipy.stats.pearsonr(combinedS,combinedCa)
ax[4].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+' (inter-Mammalia)',fontsize=fontSize)
ax[4].set_xticks([0,2,4,6,8])

# S vs. C
ax[5].plot(mammals['Cervical']-0.1*np.ones(len(mammals)),mammals['Sacral'],'o',alpha=0.5,color=colorWheel[0],markersize=markerSize)
# plot Choloepus genus from buchholtz
ax[5].plot(buchholtz['C'][buchholtz['genus']=='Choloepus'],buchholtz['S'][buchholtz['genus']=='Choloepus'],'s',alpha=0.5,color=colorWheel[5],markersize=markerSize)
# plot Bradypus genus from buchholtz
ax[5].plot(buchholtz['C'][buchholtz['genus']=='Bradypus'],buchholtz['S'][buchholtz['genus']=='Bradypus'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the one Bradypus sloth we have
ax[5].plot(sloth['Cervical'],sloth['Sacral'],'s',alpha=0.5,color=colorWheel[6],markersize=markerSize)
# plot the cetaceans from buchholtz
ax[5].plot(cetaceansAveragedSpecies['C']+0.1*np.ones(len(cetaceansAveragedSpecies)),cetaceansAveragedSpecies['S'],'^',alpha=0.5,color=colorWheel[10],markersize=markerSize+1)
# plot the manatees from various sources
ax[5].plot(manatee['C'][manatee['Species']=='Trichechus manatus'],manatee['S'][manatee['Species']=='Trichechus manatus'],'>',color=colorWheel[8],alpha=0.5,markersize=markerSize+1)
ax[5].set_xlabel('Cervical',fontsize=fontSize)
ax[5].set_ylabel('Sacral',fontsize=fontSize)
ax[5].set_xticks([5,6,7,8,9,10])
r,p = scipy.stats.pearsonr(combinedC,combinedS)
rSloths,pSloths = scipy.stats.pearsonr(buchholtz['C'],buchholtz['S'])
ax[5].set_title('r='+str(round(r,2))+', p='+"{:.2e}".format(p)+' (inter-Mammalia)\nr='+str(round(rSloths,2))+', p='+"{:.2e}".format(pSloths)+' (intra-Folivora)',fontsize=fontSize)

# subplot labels
ax[0].text(-0.2,1.05,'A',fontsize=fontSize+4,fontweight='normal',transform=ax[0].transAxes)
ax[0].text(-0.2,0.25,'B',fontsize=fontSize+4,fontweight='normal',transform=ax[0].transAxes)
ax[1].text(-0.2,1.05,'C',fontsize=fontSize+4,fontweight='normal',transform=ax[1].transAxes)
ax[2].text(-0.2,1.05,'D',fontsize=fontSize+4,fontweight='normal',transform=ax[2].transAxes)
ax[3].text(-0.2,1.05,'E',fontsize=fontSize+4,fontweight='normal',transform=ax[3].transAxes)
ax[4].text(-0.2,1.05,'F',fontsize=fontSize+4,fontweight='normal',transform=ax[4].transAxes)
ax[5].text(-0.2,1.05,'G',fontsize=fontSize+4,fontweight='normal',transform=ax[5].transAxes)

# save
plt.savefig(outputPath+'plots/extendedDataFigIntraMammals_preSketches_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(outputPath+'plots/extendedDataFigIntraMammals_preSketches_v2.pdf',dpi=300,bbox_inches='tight')