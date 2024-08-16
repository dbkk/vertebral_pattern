# make a plot of the vertebral formulas for visualization
# Fig. 1 in the paper
# and Ext. Data Fig. 2
# v2

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath
outputPath = inputPath

#%% import libraries

import pandas as pd
import numpy as np
import os
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
    "#17becf", #teal
]
colorWheel_CB = ['#000000', '#E69F00', '#56B4E9',
                  '#009E73', '#F0E442', '#0072B2',
                  '#D55E00', '#CC79A7'] # https://yoshke.org/blog/colorblind-friendly-diagrams
import scipy.stats
markerWheel = ['o','s','^','>','<','v','d'] # pour convenience
fontSize = 14
faceColor = 'white'
markerSize = 0.5
lineWidth = 0.5
dashedLineWidth = 1
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

#%% check if there is a plots directory on the outputPath, and if not, make one

savePath = outputPath + 'plots/'
if not os.path.exists(savePath):
    os.makedirs(savePath)

#%% load vertebral formula data

# load the only .csv file in the inputPath, so search for it with glob
dataPath = glob.glob(inputPath+'vertebralFormulaOrdered_v2.csv')[0]
vertebralFormula = pd.read_csv(dataPath)
vertebralFormula.rename(columns={"Cervical":"cervical","Thoracic":"thoracic","Lumbar":"lumbar","Sacral":"sacral","Caudal":"caudal"},inplace=True)

#%% separate each vertebral category into separate subplots (5)


fig,axs = plt.subplots(1,5,figsize=(3,8))
ax = plt.gca()
alphaValue = 0.7
axs[0].plot(vertebralFormula['cervical'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,label='C',linewidth=3)
axs[1].plot(vertebralFormula['thoracic'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[0],alpha=alphaValue,label='T',linewidth=3)
axs[2].plot(vertebralFormula['lumbar'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[1],alpha=alphaValue,label='L',linewidth=3)
axs[3].plot(vertebralFormula['sacral'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[2],alpha=alphaValue,label='S',linewidth=3)
axs[4].plot(vertebralFormula['caudal'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[3],alpha=alphaValue,label='Ca',linewidth=3)

# remove yticks and set ylim to be the same for all
for i in range(5):
    axs[i].set_yticks([])
    axs[i].set_ylim(1,len(vertebralFormula))

# set the x-axis labels to oscillate between the top and bottom (so on bottom for the first, top for the second, and back again, etc.)
for i in range(1,5):
    if i%2 == 0:
        axs[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,labelsize=fontSize-2)
    else:
        axs[i].tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True,labelsize=fontSize-2)

# set xlim to be 0 to 100 for thoracic and caudal
axs[1].set_xlim(-5,100)
axs[4].set_xlim(-5,100)
# and set xlim to be -5 to max + 5 for others
axs[0].set_xlim(-5,vertebralFormula['cervical'].max()+5)
axs[2].set_xlim(-5,vertebralFormula['lumbar'].max()+5)
axs[3].set_xlim(-5,vertebralFormula['sacral'].max()+5)

# set it to be top and bottom for just axs[0]
axs[0].tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
axT = axs[0].twiny()
# set the ticks on 'C' on the bottom to be 7, and 25
axs[0].set_xticks((7,25))
axs[0].set_xticklabels((7,25),fontsize=fontSize-2)
# and on the top to be 1
axT.set_xlim(-5,vertebralFormula['cervical'].max()+5)
axT.set_xticks((1,15))
axT.set_xticklabels((1,15),fontsize=fontSize-2)
# and make a vertical dashed line at 1 and 7
axs[0].plot([1,1],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)
axs[0].plot([7,7],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

# plot the x-axis labels
axs[0].set_xlabel('C',fontsize=fontSize-2)
axs[1].set_xlabel('T',fontsize=fontSize-2, labelpad=22.5)
axs[2].set_xlabel('L',fontsize=fontSize-2)
axs[3].set_xlabel('S',fontsize=fontSize-2, labelpad=22.5)
axs[4].set_xlabel('Ca',fontsize=fontSize-2)

plt.savefig(savePath+'vertebralTree_vertebralFormulae_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_vertebralFormulae_v2.pdf',dpi=300,bbox_inches='tight')


#%% separate each vertebral category into separate subplots (5) (tall version for SI)


fig,axs = plt.subplots(1,5,figsize=(3,16))
ax = plt.gca()
alphaValue = 0.7
axs[0].plot(vertebralFormula['cervical'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,label='C',linewidth=3)
axs[1].plot(vertebralFormula['thoracic'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[0],alpha=alphaValue,label='T',linewidth=3)
axs[2].plot(vertebralFormula['lumbar'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[1],alpha=alphaValue,label='L',linewidth=3)
axs[3].plot(vertebralFormula['sacral'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[2],alpha=alphaValue,label='S',linewidth=3)
axs[4].plot(vertebralFormula['caudal'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[3],alpha=alphaValue,label='Ca',linewidth=3)

# remove yticks and set ylim to be the same for all
for i in range(5):
    axs[i].set_yticks([])
    axs[i].set_ylim(1,len(vertebralFormula))

# set the x-axis labels to oscillate between the top and bottom (so on bottom for the first, top for the second, and back again, etc.)
for i in range(1,5):
    if i%2 == 0:
        axs[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,labelsize=fontSize-2)
    else:
        axs[i].tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True,labelsize=fontSize-2)

# set xlim to be 0 to 100 for thoracic and caudal
axs[1].set_xlim(-5,100)
axs[4].set_xlim(-5,100)
# and set xlim to be -5 to max + 5 for others
axs[0].set_xlim(-5,vertebralFormula['cervical'].max()+5)
axs[2].set_xlim(-5,vertebralFormula['lumbar'].max()+5)
axs[3].set_xlim(-5,vertebralFormula['sacral'].max()+5)

# set it to be top and bottom for just axs[0]
axs[0].tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
axT = axs[0].twiny()
# set the ticks on 'C' on the bottom to be 7, and 25
axs[0].set_xticks((7,25))
axs[0].set_xticklabels((7,25),fontsize=fontSize-2)
# and on the top to be 1
axT.set_xlim(-5,vertebralFormula['cervical'].max()+5)
axT.set_xticks((1,15))
axT.set_xticklabels((1,15),fontsize=fontSize-2)
# and make a vertical dashed line at 1 and 7
axs[0].plot([1,1],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)
axs[0].plot([7,7],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

# plot the x-axis labels
axs[0].set_xlabel('C',fontsize=fontSize-2)
axs[1].set_xlabel('T',fontsize=fontSize-2, labelpad=22.5)
axs[2].set_xlabel('L',fontsize=fontSize-2)
axs[3].set_xlabel('S',fontsize=fontSize-2, labelpad=22.5)
axs[4].set_xlabel('Ca',fontsize=fontSize-2)

plt.savefig(savePath+'vertebralTree_vertebralFormulae_tall_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_vertebralFormulae_tall_v2.pdf',dpi=300,bbox_inches='tight')


#%% make a plot of some C+S

fontSize = 12
params = {
   'axes.labelsize': fontSize,
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
   }
mpl.rcParams.update(params)

fig = plt.figure(figsize=(1.5,8))
ax = plt.gca()
alphaValue = 0.7
plt.plot(vertebralFormula['cervical']+vertebralFormula['sacral'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[0],alpha=alphaValue,linewidth=3,linestyle='-',label='C+S')
plt.plot(vertebralFormula['cervical']-vertebralFormula['sacral'],np.flipud(np.arange(len(vertebralFormula))),color=colorWheel[1],alpha=alphaValue,linewidth=3,linestyle='-',label='C-S')
plt.plot(np.zeros(len(vertebralFormula)),np.flipud(np.arange(len(vertebralFormula))),'k--',linewidth=dashedLineWidth)
plt.legend(frameon=False,loc=[0.27,0.85],handlelength=1.0,fontsize=fontSize+0)
plt.ylim(1,len(vertebralFormula))
plt.xlim(-10,50)
plt.xlabel('C+S, C-S',fontsize=fontSize+0)
plt.xticks((0,25,50))
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks([])

axT = ax.twiny()
axT.set_xticks((0,25,50))
axT.set_xlim(-10,50)

plt.savefig(savePath+'vertebralTree_CS_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_CS_v2.pdf',dpi=300,bbox_inches='tight')

#%% make a plot of some non-trivial constraints: C+T-S-Ca

fontSize = 12
params = {
   'axes.labelsize': fontSize,
   'font.family': fontToUse,
   'font.style': 'normal',
   'font.weight': 'normal',
   'text.usetex': False,
   'font.size': fontSize,
   'xtick.labelsize': fontSize,
   'ytick.labelsize': fontSize,
   'text.usetex': False,
   }
mpl.rcParams.update(params)

fig = plt.figure(figsize=(1.5,8))
ax = plt.gca()
alphaValue = 0.7
plt.plot(vertebralFormula['cervical']+vertebralFormula['thoracic']-vertebralFormula['sacral']-vertebralFormula['caudal'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=1.0,label='cervical',linewidth=3)
plt.plot(np.zeros(len(vertebralFormula)),np.flipud(np.arange(len(vertebralFormula))),'k--',linewidth=dashedLineWidth)
plt.ylim(1,len(vertebralFormula))
plt.xlim(-60,60)
plt.xlabel('C+T-S-Ca',fontsize=fontSize+0)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks([])

axT = ax.twiny()
axT.set_xlim(-60,60)

plt.savefig(savePath+'vertebralTree_CTSCa_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_CTSCa_v2.pdf',dpi=300,bbox_inches='tight')

# make plots of mammalian special constraints
fontSize = 14

#%% C+T

fig = plt.figure(figsize=(1.5,16))
ax = plt.gca()
alphaValue = 1
plt.plot(vertebralFormula['cervical']+vertebralFormula['thoracic'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,linewidth=3,linestyle='-',label='C+T')
# find mean of C+T excluding Amphibians and snakes (T > 100)
meanCT = np.nanmean(vertebralFormula['cervical'][(vertebralFormula['thoracic']<100) & (vertebralFormula['Class']!='Amphibia')] + vertebralFormula['thoracic'][(vertebralFormula['thoracic']<100) & (vertebralFormula['Class']!='Amphibia')])
plt.ylim(1,len(vertebralFormula))
plt.xlim(0,40)
plt.xlabel('C+T',fontsize=fontSize-2)
plt.xticks((0,int(meanCT),40))
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=fontSize-2)
plt.yticks([])

# plot a vertical dashed line at int(meanCT)
plt.plot([int(meanCT),int(meanCT)],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

axT = ax.twiny()
axT.set_xticks((0,int(meanCT),40))
axT.set_xticklabels((0,int(meanCT),40),fontsize=fontSize-2)
axT.set_xlim(0,40)

plt.savefig(savePath+'vertebralTree_CT_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_CT_v2.pdf',dpi=300,bbox_inches='tight')


#%% T+L

fig = plt.figure(figsize=(1.5,16))
ax = plt.gca()
alphaValue = 1
plt.plot(vertebralFormula['thoracic']+vertebralFormula['lumbar'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,linewidth=3,linestyle='-',label='T+L')
plt.plot(np.zeros(len(vertebralFormula)),np.flipud(np.arange(len(vertebralFormula))),'k--',linewidth=dashedLineWidth)
plt.ylim(1,len(vertebralFormula))
plt.xlim(0,100)
plt.xlabel('T+L',fontsize=fontSize-2)
plt.xticks((0,20,50,100))
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=fontSize-2)
plt.yticks([])

# plot a vertical dashed line at 20
plt.plot([20,20],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

axT = ax.twiny()
axT.set_xticks((0,20,50,100))
axT.set_xticklabels((0,20,50,100),fontsize=fontSize-2)
axT.set_xlim(0,100)

plt.savefig(savePath+'vertebralTree_TL_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_TL_v2.pdf',dpi=300,bbox_inches='tight')


#%% S+L

fig = plt.figure(figsize=(1.5,16))
ax = plt.gca()
alphaValue = 1
plt.plot(vertebralFormula['sacral']+vertebralFormula['lumbar'],np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,linewidth=3,linestyle='-',label='T+L')
plt.plot(np.zeros(len(vertebralFormula)),np.flipud(np.arange(len(vertebralFormula))),'k--',linewidth=dashedLineWidth)
plt.ylim(1,len(vertebralFormula))
plt.xlim(-2,25)
plt.xlabel('L+S',fontsize=fontSize-2)
plt.xticks((0,9,25))
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=fontSize-2)
plt.yticks([])

# plot a vertical dashed line at 9
plt.plot([9,9],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

axT = ax.twiny()
axT.set_xticks((0,9,25))
axT.set_xticklabels((0,9,25),fontsize=fontSize-2)
axT.set_xlim(-2,25)

plt.savefig(savePath+'vertebralTree_SL_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_SL_v2.pdf',dpi=300,bbox_inches='tight')


#%% S+Ca

fig = plt.figure(figsize=(1.5,16))
ax = plt.gca()
alphaValue = 1
plt.plot(vertebralFormula['sacral']+vertebralFormula['caudal']/3,np.flipud(np.arange(len(vertebralFormula))),color='black',alpha=alphaValue,linewidth=3,linestyle='-',label='T+L')
plt.plot(np.zeros(len(vertebralFormula)),np.flipud(np.arange(len(vertebralFormula))),'k--',linewidth=dashedLineWidth)
plt.ylim(1,len(vertebralFormula))
plt.xlim(-2,50)
plt.xlabel('S+Ca/3',fontsize=fontSize-2)
plt.xticks((0,10,25,50))
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True, labelsize=fontSize-2)
plt.yticks([])

# plot a vertical dashed line at 10
plt.plot([10,10],[0,len(vertebralFormula)],'k--',linewidth=dashedLineWidth)

axT = ax.twiny()
axT.set_xticks((0,10,25,50))
axT.set_xticklabels((0,10,25,50),fontsize=fontSize-2)
axT.set_xlim(-2,50)

plt.savefig(savePath+'vertebralTree_SCa_v2.png',dpi=300,bbox_inches='tight')
plt.savefig(savePath+'vertebralTree_SCa_v2.pdf',dpi=300,bbox_inches='tight')