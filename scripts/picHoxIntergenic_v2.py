# perform PIC on the intergenic distances as well as their corresponding vertebral data
# v2
# now also saves the ancestral reconstruction

#%% set the paths

basePath = './'
scriptPath = basePath+'scripts/'
inputPath = basePath +'hoxData/'
inputPath2 = basePath + 'intergenic/'
outputPath = basePath

#%% import libraries and some other preliminaries

from skbio import TreeNode
from io import StringIO
import pandas as pd
import copy
import numpy as np
import os

# percent threshold for getting rid of columns with NaNs
N = 20

# remove snakes?
removeSnakes = True

# make the path outputPath + 'intergenic' if it doesn't exist
if not os.path.exists(outputPath+'intergenic'):
    os.mkdir(outputPath+'intergenic')
# and to that add intergenicIndividual
if not os.path.exists(outputPath+'intergenic/intergenicIndividual'):
    os.mkdir(outputPath+'intergenic/intergenicIndividual')
    

def picFunction(tree,x,tipNamesOrig,cols,speciesKey,nameKey,classKey):
    
    # first reduce the tree and x to the species in x which have non-nan data for given columns
    tree,x,tipNamesOrig = reduceTree(tree,x,cols,speciesKey)
    
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
        picNames = ([nameBank[speciesBank.index(picSpecies[0])],nameBank[speciesBank.index(picSpecies[1])]])
        treeTimes = []
        treeTimes.append(float(tempStr[len(picSpecies[0])+1:tempStr.find(',')]))
        treeTimes.append(float(tempStr2[len(picSpecies[1])+1:]))
        
        # get the classes
        # picClasses = ([xPICdf['class'][xPICdf[speciesKey]==picSpecies[0]].iloc[0],xPICdf['class'][xPICdf[speciesKey]==picSpecies[1]].iloc[0]])
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
    
    # make the ancestral states
    xAnc = pd.DataFrame()
    xAnc[nameKey] = nameBank
    xAnc[speciesKey] = speciesBank
    xAnc[classKey] = classBank
    newCols3 = []
    for i in range(len(xKeys)):
        newCols3.append(pd.Series(xBank[:,i], name=xKeys[i], index=xAnc.index))
    xAnc = pd.concat([xAnc,*newCols3],axis=1)
    
    # count the number in each class (for the ancestral states)
    # get the node corresponding to each ancestral state by looking at the tree
    classes = list(set(xAnc[classKey].to_list()[0:len(tipNamesOrig)]))
    classNumbers = np.zeros((len(xAnc),len(classes)))
    nodeList = []
    for i in range(len(xAnc)):
        node = tree.lowest_common_ancestor(str(xAnc[speciesKey].iloc[i]).split('+')).name
        # if node is type 'NoneType', then set node = 0
        if node == None:
            node = 0
        elif node.isdigit():
            node = int(node)
        else:
            node = 0
        nodeList.append(node)
        classTemp = str(xAnc[classKey].iloc[i]).split('+')
        for j in range(len(classes)):
            classNumbers[i,j] = classTemp.count(classes[j])
    # in this implementation the last node is the root
    nodeList[-1] = 'root'
    xAnc['node'] = nodeList
    for i in range(len(classes)):
        xAnc[classes[i]] = classNumbers[:,i].astype(int)
    
    return xPICn,xPIC,xAnc

#%% reduce tree and x to the species in x which have non-nan data for given columns

def reduceTree(tree,x,cols,speciesKey):

    indColumns = list([0,1,2]) # we want to include the name, species, and class columns
    for i in range(len(cols)):
        indColumns = np.append(indColumns,np.array(list(x.columns).index(cols[i])))
    indColumns = np.array(indColumns).astype(int)
    
    x2 = x.iloc[:,indColumns]
    x2 = x2.dropna()
    x2 = x2.reset_index(drop=True)
    
    # shear the tree
    tree2 = tree.shear(x2[speciesKey].values.tolist())
    tree2.prune()
    
    # get the new tip names
    tipNamesOrig2 = []
    for node in tree2.tips():
        tipNamesOrig2.append(node.name)
    tipNamesOrig2 = np.array(tipNamesOrig2)
    
    return tree2,x2,tipNamesOrig2

#%% load the species list that we uploaded to time tree

treeFileName = inputPath+'hoxIntergenicTree.nwk'

with open(treeFileName) as f:
    treeFile = f.read()
tree = TreeNode.read(StringIO(treeFile))

# species in tree
tipNamesOrig = []
for node in tree.tips():
    tipNamesOrig.append(node.name)

#%% load the data to be run through the PIC

dataFileName = inputPath+'speciesWithGenomeIntergenicDistances_v2.csv'
x = pd.read_csv(dataFileName)

#%% reduce the tree to the species in x
# get the keys and see if it is "Species" or "species"

xKeys = list(x.keys())
if 'Species' in xKeys:
    speciesKey = 'Species'
elif 'species' in xKeys:
    speciesKey = 'species'
else:
    print('ERROR: no species key found in data!')
    exit()
    
nameKey = 'Common Name'
classKey = 'Class'

# # list of species to remove from the tree and x
if removeSnakes:
    speciesToRemove = (['Crotalus tigris',
                        'Thamnophis elegans',
                        'Pantherophis guttatus'])
else:
    speciesToRemove = ([])
# remove from x
indKeep = []
for i in range(len(x)):
    if x[speciesKey].iloc[i] not in speciesToRemove:
        indKeep.append(i)
x = x.iloc[indKeep,:]
# reset index
x = x.reset_index(drop=True)

tree = tree.shear(x[speciesKey].values.tolist())
tree.prune()

# double check that the number is the same now:
tipNamesOrig = []
for node in tree.tips():
    tipNamesOrig.append(node.name)
tipNamesOrig = np.array(tipNamesOrig)
if len(tipNamesOrig) != len(x[speciesKey].values):
    print('ERROR: tree and data do not have the same number of species!')
    exit()
else:
    print('Tree and data have the same number of species! (They may not still match of course!)')

#%% get the indCols for the intergenic distances

indCols = np.arange(xKeys.index('A1'),xKeys.index('D13')+1)
dataCols = np.array(xKeys)[indCols]

#%% find those columns in x which only have NaNs

indColsNans = []
for i in range(len(dataCols)):
    if x[dataCols[i]].isna().all():
        indColsNans.append(i)
indColsNans = np.array(indColsNans).astype(int)
# remove these columns from indCols and dataCols
indCols = np.delete(indCols,indColsNans)
dataCols = np.delete(dataCols,indColsNans)

#%% find those columns which have more than N% NaNs

indColsNans = []
for i in range(len(dataCols)):
    if x[dataCols[i]].isna().sum()/len(x) > N/100:
        indColsNans.append(i)
indColsNans = np.array(indColsNans).astype(int)
# remove these columns from indCols and dataCols
indCols = np.delete(indCols,indColsNans)
dataCols = np.delete(dataCols,indColsNans)

#%% run the PIC with intergenic data paired with a certain vertebral data combination

vertebralDataToTest = (['Cervical','Thoracic','Lumbar','Sacral','Caudal'])

for i in range(len(vertebralDataToTest)):
    print(vertebralDataToTest[i])
    # temporarily keep only those species which have no NaN for this vertebral data
    indKeep = []
    for j in range(len(x)):
        if ~np.isnan(x[vertebralDataToTest[i]].iloc[j]):
            indKeep.append(j)
    xTemp = x.iloc[indKeep,:]
    # only 
    indColsTemp = np.append(xTemp.columns.get_loc(vertebralDataToTest[i]),indCols)
    xTemp = xTemp.iloc[:,np.append(np.array([0,1,2]),indColsTemp)]
    dataColsTemp = np.append(vertebralDataToTest[i],dataCols)
    
    # shear the tree and get the new tipNamesOrig by reducing the species in xTemp
    treeTemp,xTempTemp,tipNamesOrigTemp = reduceTree(tree,xTemp,dataColsTemp,speciesKey)
    
    xPICn,xPIC,xAnc = picFunction(treeTemp,xTempTemp,tipNamesOrigTemp,dataColsTemp,speciesKey,nameKey,classKey)
    if removeSnakes:
        xPIC.to_csv(outputPath+'intergenic/pic_raw_intergenic_'+vertebralDataToTest[i]+'_noSnakes.csv',index=False)
        xPICn.to_csv(outputPath+'intergenic/pic_normalized_intergenic_'+vertebralDataToTest[i]+'_noSnakes.csv',index=False)
        xAnc.to_csv(outputPath+'intergenic/ancestralStates_intergenic_'+vertebralDataToTest[i]+'_noSnakes.csv',index=False)
    else:
        xPIC.to_csv(outputPath+'intergenic/pic_raw_intergenic_'+vertebralDataToTest[i]+'.csv',index=False)
        xPICn.to_csv(outputPath+'intergenic/pic_normalized_intergenic_'+vertebralDataToTest[i]+'.csv',index=False)
        xAnc.to_csv(outputPath+'intergenic/ancestralStates_intergenic_'+vertebralDataToTest[i]+'.csv',index=False)

    # also go through each of the remaining hox genes separately
    for j in range(1,len(dataColsTemp)):
        dataColsTempTemp = np.array([dataColsTemp[0],dataColsTemp[j]])
        treeTempTemp,xTempTemp,tipNamesOrigTempTemp = reduceTree(tree,xTemp,dataColsTempTemp,speciesKey)
        
        xPICn2,xPIC2,xAnc2 = picFunction(treeTempTemp,xTempTemp,tipNamesOrigTempTemp,dataColsTempTemp,speciesKey,nameKey,classKey)
        if removeSnakes:
            xPIC2.to_csv(outputPath+'intergenic/intergenicIndividual/pic_raw_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'_noSnakes.csv',index=False)
            xPICn2.to_csv(outputPath+'intergenic/intergenicIndividual/pic_normalized_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'_noSnakes.csv',index=False)
            xAnc2.to_csv(outputPath+'intergenic/intergenicIndividual/ancestralStates_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'_noSnakes.csv',index=False)   
        else:
            xPIC2.to_csv(outputPath+'intergenic/intergenicIndividual/pic_raw_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'.csv',index=False)
            xPICn2.to_csv(outputPath+'intergenic/intergenicIndividual/pic_normalized_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'.csv',index=False)
            xAnc2.to_csv(outputPath+'intergenic/intergenicIndividual/ancestralStates_intergenic_'+vertebralDataToTest[i]+'_'+dataColsTemp[j]+'.csv',index=False)
