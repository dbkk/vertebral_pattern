# R script to determine the ancestral state reconstruction using phytools' contMap and fastAnc functions
# outputs the nodes with the predicted ancestral vertebral formula
# outputs the trees with color corresponding to the specific predicted vertebral count or combination thereof

# libraries

library(ape)
library(phytools)
library(scico)

# set path
basePath <- "./"
savePath <- paste(basePath, "plots/treesWithFastAnc/", sep="")
# make this directory if it doesn't already exist
dir.create(savePath, showWarnings = FALSE)

# load the phylogenetic tree
phy_tree <- read.tree(paste(basePath, "vertebralTree.nwk", sep=""))
phy_tree$node.label[1] <- "full"

# load all vertebral data from the .csv file
vertebralData <- read.csv(paste(basePath, "vertebralFormulaOrdered_v2.csv", sep=""), header = TRUE)
species_lists <- split(vertebralData$Species, vertebralData$Class)
species_lists$Amphibia <- gsub(" ", "_", species_lists$Amphibia)
species_lists$Mammalia <- gsub(" ", "_", species_lists$Mammalia)
species_lists$Reptilia <- gsub(" ", "_", species_lists$Reptilia)
species_lists$Aves <- gsub(" ", "_", species_lists$Aves)

# set the character data to simply be the Species", "Cervical", "Thoracic", "Lumbar", "Sacral", "Caudal" columns of the vertebral data
character_data <- vertebralData[, c("Species", "Cervical", "Thoracic", "Lumbar", "Sacral", "Caudal")]

# replace spaces with underscores in Species column
character_data$Species <- gsub(" ", "_", character_data$Species)

# identify species to prune (because the relative branch time is zero)
species_to_prune <- "Cygnus_atratus"

# remove the species from character_data
character_data <- character_data[character_data$Species != species_to_prune, ]

# prune the species from phy_tree
phy_tree <- drop.tip(phy_tree, species_to_prune)

# check for matching species names
matching_species <- intersect(phy_tree$tip.label, character_data$Species)

# filter character_data and phy_tree to only include matching species
character_data <- character_data[character_data$Species %in% matching_species, ]
phy_tree <- drop.tip(phy_tree, phy_tree$tip.label[!phy_tree$tip.label %in% matching_species])

# check species order again after filtering
species_order <- match(phy_tree$tip.label, character_data$Species)
cervical <- character_data[["Cervical"]][species_order]
thoracic <- character_data[["Thoracic"]][species_order]
lumbar <- character_data[["Lumbar"]][species_order]
sacral <- character_data[["Sacral"]][species_order]
caudal <- character_data[["Caudal"]][species_order]

# create a new data frame with species names and cervical data
data_for_analysis <- data.frame(
  Species = character_data$Species[species_order],  # ensure order matches phy_tree$tip.label
  Cervical = character_data$Cervical[species_order],  # adjust with actual trait column name
  Thoracic = character_data$Thoracic[species_order],  # adjust with actual trait column name
  Lumbar = character_data$Lumbar[species_order],  # adjust with actual trait column name
  Sacral = character_data$Sacral[species_order],  # adjust with actual trait column name
  Caudal = character_data$Caudal[species_order],  # adjust with actual trait column name
  stringsAsFactors = FALSE  # ensure species names are treated as characters, not factors
)

# set names for the cervical data vector
cervical <- setNames(data_for_analysis$Cervical, data_for_analysis$Species)
thoracic <- setNames(data_for_analysis$Thoracic, data_for_analysis$Species)
lumbar <- setNames(data_for_analysis$Lumbar, data_for_analysis$Species)
sacral <- setNames(data_for_analysis$Sacral, data_for_analysis$Species)
caudal <- setNames(data_for_analysis$Caudal, data_for_analysis$Species)

# perform ancestral state reconstruction
anc_states_cervical <- fastAnc(phy_tree,cervical,vars=TRUE,CI=TRUE)
anc_states_thoracic <- fastAnc(phy_tree,thoracic,vars=TRUE,CI=TRUE)
anc_states_lumbar <- fastAnc(phy_tree,lumbar,vars=TRUE,CI=TRUE)
anc_states_sacral <- fastAnc(phy_tree,sacral,vars=TRUE,CI=TRUE)
anc_states_caudal <- fastAnc(phy_tree,caudal,vars=TRUE,CI=TRUE)

# extract the node numbers and ancestral state estimates
new_node_numbers <- names(anc_states_cervical$ace)

# original nodes in the pruned tree
original_node_numbers <- phy_tree$node.label

# create a data frame to map new node numbers to original node numbers
node_mapping <- data.frame(
  Original_Node = original_node_numbers,
  New_Node = new_node_numbers
)

# function to remove nested quotes
remove_nested_quotes <- function(node_label) {
  gsub("'", "", node_label)
}

# create a dataframe to save
anc <- data.frame(
  old_node = sapply(node_mapping$Original_Node, remove_nested_quotes),
  new_node = node_mapping$New_Node,
  cervical = anc_states_cervical$ace,
  cervical_var = anc_states_cervical$var,
  cervical_CI95 = anc_states_cervical$CI95,
  thoracic = anc_states_thoracic$ace,
  thoracic_var = anc_states_thoracic$var,
  thoracic_CI95 = anc_states_thoracic$CI95,
  lumbar = anc_states_lumbar$ace,
  lumbar_var = anc_states_lumbar$var,
  lumbar_CI95 = anc_states_lumbar$CI95,
  sacral = anc_states_sacral$ace,
  sacral_var = anc_states_sacral$var,
  sacral_CI95 = anc_states_sacral$CI95,
  caudal = anc_states_caudal$ace,
  caudal_var = anc_states_caudal$var,
  caudal_CI95 = anc_states_caudal$CI95
)

# write combined data to CSV
write.csv(anc, file = paste(basePath, "vertebralFormula_fastAncReconstruction.csv", sep=""), row.names = FALSE)

# now plot! the contMap function actually does the node estimates using fastAnc internally!

# rotate the tree to be in the order of the manuscript Fig. 1
phy_tree_rotated <- rotateConstr(phy_tree, rev(character_data$Species))

obj_cervical<-contMap(phy_tree_rotated,setNames(data_for_analysis$Cervical, data_for_analysis$Species),plot=FALSE,lims=c(0,20))
obj_thoracic<-contMap(phy_tree_rotated,setNames(data_for_analysis$Thoracic, data_for_analysis$Species),plot=FALSE,lims=c(0,25))
obj_thoracicB<-contMap(phy_tree_rotated,setNames(data_for_analysis$Thoracic, data_for_analysis$Species),plot=FALSE,lims=c(0,250))
obj_lumbar<-contMap(phy_tree_rotated,setNames(data_for_analysis$Lumbar, data_for_analysis$Species),plot=FALSE,lims=c(0,10))
obj_sacral<-contMap(phy_tree_rotated,setNames(data_for_analysis$Sacral, data_for_analysis$Species),plot=FALSE,lims=c(0,20))
obj_caudal<-contMap(phy_tree_rotated,setNames(data_for_analysis$Caudal, data_for_analysis$Species),plot=FALSE,lims=c(0,25))
obj_caudalB<-contMap(phy_tree_rotated,setNames(data_for_analysis$Caudal, data_for_analysis$Species),plot=FALSE,lims=c(0,100))
obj_CS<-contMap(phy_tree_rotated,setNames(data_for_analysis$Cervical-data_for_analysis$Sacral, data_for_analysis$Species),plot=FALSE,lims=c(-10,10))
# get the mean value of the mammalian Thoracic plus Lumbar vertebrae using species_lists$Mammalia
# first get the indices of the mammalian species
index_mammals <- which(data_for_analysis$Species %in% species_lists$Mammalia)
mammalTL <- mean(data_for_analysis$Thoracic[index_mammals]+data_for_analysis$Lumbar[index_mammals])
obj_TL<-contMap(phy_tree_rotated,setNames(data_for_analysis$Thoracic+data_for_analysis$Lumbar - mammalTL, data_for_analysis$Species),plot=FALSE,lims=c(-20,20))
mammalLS <- mean(data_for_analysis$Lumbar[index_mammals]+data_for_analysis$Sacral[index_mammals])
obj_LS<-contMap(phy_tree_rotated,setNames(data_for_analysis$Lumbar+data_for_analysis$Sacral-mammalLS, data_for_analysis$Species),plot=FALSE,lims=c(-20,20))
mammalSCa <- mean(data_for_analysis$Sacral[index_mammals]+data_for_analysis$Caudal[index_mammals])
mammalSCab <- mean(3*data_for_analysis$Sacral[index_mammals]+data_for_analysis$Caudal[index_mammals])
obj_SCa<-contMap(phy_tree_rotated,setNames(3*data_for_analysis$Sacral+data_for_analysis$Caudal-mammalSCab, data_for_analysis$Species),plot=FALSE,lims=c(-20,20))
obj_CTSCa<-contMap(phy_tree_rotated,setNames(data_for_analysis$Cervical+data_for_analysis$Thoracic-data_for_analysis$Sacral-data_for_analysis$Caudal, data_for_analysis$Species),plot=FALSE,lims=c(-20,20))

# plot the individual vertebral counts using oslo (reversed)

obj_dataset <- list(
  cervical = obj_cervical,
  thoracic = obj_thoracic,
  thoracicB = obj_thoracicB,
  lumbar = obj_lumbar,
  sacral = obj_sacral,
  caudal = obj_caudal,
  caudalB = obj_caudalB
)

for (i in 1:length(obj_dataset)) {
  pdf(paste(savePath, names(obj_dataset)[i], "_fastAnc.pdf", sep=""))
  # if we are doing the thoracic or caudal, then factor = 2
  if (names(obj_dataset)[i] == "thoracic" | names(obj_dataset)[i] == "thoracicB" | names(obj_dataset)[i] == "caudal" | names(obj_dataset)[i] == "caudalB") {
    plot(
      setMap(obj_dataset[[i]], rev(scico(10, palette = "bamako"))),
      fsize = c(2.0, 0.7),
      leg.text = names(obj_dataset)[i],
      lwd = 1.5,
      offset = 1,
      outline = FALSE
    )
  } else {
    plot(
      setMap(obj_dataset[[i]], rev(scico(10, palette = "bamako"))),
      fsize = c(1.5, 0.7),
      leg.text = names(obj_dataset)[i],
      lwd = 1.5,
      offset = 1,
      outline = FALSE
    )
  }
  dev.off()
}

# plot some of the combinations using vik
obj_dataset_combo <- list(
  CS = obj_CS,
  TL = obj_TL,
  LS = obj_LS,
  SCa = obj_SCa,
  CTSCa = obj_CTSCa
)

for (i in 1:length(obj_dataset_combo)) {
  pdf(paste(savePath, names(obj_dataset_combo)[i], "_fastAnc.pdf", sep=""))
  plot(
    setMap(obj_dataset_combo[[i]], scico(10, palette = "vik")),
    fsize = c(1.5, 0.7),
    leg.text = names(obj_dataset_combo)[i],
    lwd = 1.5,
    offset = 1,
    outline = FALSE
  )
  dev.off()
}

# also plot narrower versions of CS and CTSCa

obj_dataset_combo_narrow <- list(
  CS = obj_CS,
  CTSCa = obj_CTSCa
)

for (i in 1:length(obj_dataset_combo_narrow)) {
  pdf(paste(savePath, names(obj_dataset_combo_narrow)[i], "_fastAnc_narrow.pdf", sep=""))
  plot(
    setMap(obj_dataset_combo_narrow[[i]], scico(10, palette = "vik")),
    fsize = c(2.0, 0.7),
    leg.text = names(obj_dataset_combo_narrow)[i],
    lwd = 1.5,
    offset = 1,
    outline = FALSE
  )
  dev.off()
}

