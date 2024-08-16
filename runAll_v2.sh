#!/bin/bash

# The following Python libraries must be pre-installed
# numpy, pandas, skbio (scikit-bio), sklearn, sympy
# And the following R libraries must be pre-installed
# ape, phangorn, phytools (may need to install phangorn first), scico (for its colormaps)

# First make the vertical plots of the vertebral categories and combinations in Fig. 1 and Extended Data Fig. 2 organized according to the tree
python scripts/makeVertebralPlots_v2.py

# Next run the vertebral constraint and plasticity analysis:

# Get the constraints for each branch of the tree (N>20)
python scripts/getVertebralConstraints_v2.py

# Do some post analysis
python scripts/getVertebralConstraints_postAnalysis_v2.py

# Run the PIC on these patterns
python scripts/picVertebral_v2.py

# Plot the results (Fig. 2, Extended Data Figs. 4,5)
python scripts/vertebralConstraints_combinePlot_v2.py

# Analyze all bird data and theropods, bats, and pterosaurs regarding bird constraint.
python scripts/analyzePlotBirdConstraint_v2.py

# Analyze the exceptional mammalians (sloths, manatees, cetaceans)
python scripts/analyzePlotExceptionalMammals_v2.py

# Next run the Hox intergenic distance vs. vertebral count analysis:

# Run the PIC on the intergenic distances and (partial) vertebral counts:
python scripts/picHoxIntergenic_v2.py

# Analyze and plot the results (Fig. 4, Extended Data Figs. 6,7)
python scripts/compareHoxIntergenicDistancesVertebralCounts_v2.py

# Additionally run the ancestral state reconstruction using phytools
Rscript scripts/ancestralReconstructionVertebral.R

# Additionally visualize the varying evolutionary rates using the (absolute) PIC values at the nodes
Rscript scripts/ancestralReconstructionVertebral.R

# Additionally run the ancestral state reconstruction using phytools
Rscript scripts/varyingRatesVertebralPIC.R

# Double check the Pearson correlation for the main vertebral and intergenic figures after PIC
Rscript scripts/vertebralCheckPIC.R
Rscript scripts/intergenicCheckPIC.R
