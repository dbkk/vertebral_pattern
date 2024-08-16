# R Script to plot the PIC contrasts on the tree to get a picture of the varying rates of evolution
# outputs the trees with colored nodes corresponding to the absolute PIC value
# large values mean large rates of change
# also prints out the nodes with the top PIC values

# load libraries
library(ape)
library(phytools)
library(scico)
library(phangorn)

# set path
basePath <- "./"
savePath <- paste(basePath, "plots/treesWithPIC/", sep="")
# make this directory if it doesn't already exist
dir.create(savePath, showWarnings = FALSE)

# load the phylogenetic tree
phy_tree_original <- read.tree(paste(basePath, "vertebralTree.nwk", sep=""))
phy_tree_original$node.label[1] <- "full"

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

# identify species to prune
species_to_prune <- "Cygnus_atratus"

# remove the species from character_data (because the relative branch time is zero)
character_data <- character_data[character_data$Species != species_to_prune, ]

# prune the species from phy_tree
phy_tree <- drop.tip(phy_tree_original, species_to_prune)

# check for matching species names
matching_species <- intersect(phy_tree$tip.label, character_data$Species)

# filter character_data and phy_tree to only include matching species (double check)
character_data <- character_data[character_data$Species %in% matching_species, ]
phy_tree <- drop.tip(phy_tree, phy_tree$tip.label[!phy_tree$tip.label %in% matching_species])

# rotate the tree to be like Fig. 1 in the main manuscript
phy_tree <- rotateConstr(phy_tree, rev(character_data$Species))

# check species order again after filtering and rotating
species_order <- match(phy_tree$tip.label, character_data$Species)
cervical <- character_data[["Cervical"]][species_order]
thoracic <- character_data[["Thoracic"]][species_order]
lumbar <- character_data[["Lumbar"]][species_order]
sacral <- character_data[["Sacral"]][species_order]
caudal <- character_data[["Caudal"]][species_order]

# create a new data frame with species names and cervical data
data_for_analysis <- data.frame(
  Species = character_data$Species[species_order],  # Ensure order matches phy_tree$tip.label
  Cervical = character_data$Cervical[species_order],  # Adjust with your actual trait column name
  Thoracic = character_data$Thoracic[species_order],  # Adjust with your actual trait column name
  Lumbar = character_data$Lumbar[species_order],  # Adjust with your actual trait column name
  Sacral = character_data$Sacral[species_order],  # Adjust with your actual trait column name
  Caudal = character_data$Caudal[species_order],  # Adjust with your actual trait column name
  stringsAsFactors = FALSE  # Ensure species names are treated as characters, not factors
)

# set names for the vertebral data
cervical <- setNames(data_for_analysis$Cervical, data_for_analysis$Species)
thoracic <- setNames(data_for_analysis$Thoracic, data_for_analysis$Species)
lumbar <- setNames(data_for_analysis$Lumbar, data_for_analysis$Species)
sacral <- setNames(data_for_analysis$Sacral, data_for_analysis$Species)
caudal <- setNames(data_for_analysis$Caudal, data_for_analysis$Species)

# calculate the phylogenetic independent contrasts for different traits
pic_cervical <- pic(cervical, phy_tree, scaled = TRUE)
pic_thoracic <- pic(thoracic, phy_tree, scaled = TRUE)
pic_lumbar <- pic(lumbar, phy_tree, scaled = TRUE)
pic_sacral <- pic(sacral, phy_tree, scaled = TRUE)
pic_caudal <- pic(caudal, phy_tree, scaled = TRUE)
pic_CS <- pic(cervical-sacral, phy_tree, scaled = TRUE)
pic_CTSCa <- pic(cervical+thoracic-sacral-caudal, phy_tree, scaled = TRUE)
pic_CT <- pic(cervical+thoracic, phy_tree, scaled = TRUE)
pic_TL <- pic(thoracic+lumbar, phy_tree, scaled = TRUE)
pic_LS <- pic(lumbar+sacral, phy_tree, scaled = TRUE)
pic_SCa <- pic(sacral+caudal, phy_tree, scaled = TRUE)
pic_SCaB <- pic(3*sacral+caudal+caudal, phy_tree, scaled = TRUE)

# make a dataframe with all of these and then loop through them, plot them, and save the plots using the names
pic_values_dataframe <- data.frame(
  pic_cervical = pic_cervical,
  pic_thoracic = pic_thoracic,
  pic_lumbar = pic_lumbar,
  pic_sacral = pic_sacral,
  pic_caudal = pic_caudal,
  pic_CS = pic_CS,
  pic_CTSCa = pic_CTSCa,
  pic_CT = pic_CT,
  pic_TL = pic_TL,
  pic_LS = pic_LS,
  pic_SCa = pic_SCa,
  pic_SCaB = pic_SCaB
)

# write this to a .csv file
write.csv(pic_values_dataframe, file = paste(basePath, "vertebralFormula_varyingRatesPIC.csv", sep=""), row.names = FALSE)

# plot the PIC contrasts on the tree at the different nodes

# define a color palette
scico_palette <- scico(100, palette = "oslo")
# reverse the color palette if desired
scico_palette <- rev(scico_palette)

# function to add a custom color bar legend
add_color_bar <- function(colors, min_val, max_val, title = "", num_ticks = 5, tick_cex = 0.8) {
  # par(mar = c(5, 4, 4, 2) + 0.1, xpd = TRUE)
  legend_ticks <- seq(min_val, max_val, length.out = num_ticks)
  legend_labels <- round(legend_ticks, 2)
  
  # create a blank plot for the legend
  plot(c(0, 0.5), c(0, 1), type = "n", xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n")
  
  # add the color bar
  rect(0, seq(0, 1, length.out = length(colors) + 1)[-length(colors)],
       0.1, seq(1 / length(colors), 1, length.out = length(colors)),
       col = colors, border = NA)
  
  # add the axis labels
  axis(4, at = seq(0, 1, length.out = num_ticks), labels = legend_labels, las = 1, cex.axis = tick_cex)
  mtext(title, side = 4, line = 3, cex = tick_cex)
}

# convert the tree to a string
phy_tree_string <- write.tree(phy_tree)
phy_tree_phangorn <- read.tree(text = phy_tree_string)

# loop through and plot
for (pic_name in colnames(pic_values_dataframe)) {
  # take the absolute value of the PIC values
  pic_values <- abs(pic_values_dataframe[[pic_name]])
  # take the log and take care of the zero
  pic_values <- log(pic_values + 1)
  pic_range <- range(pic_values)
  pic_normalized <- (pic_values - pic_range[1]) / diff(pic_range) # normalize for plotting

  # map normalized PIC values to colors
  node_colors <- scico_palette[as.numeric(cut(pic_normalized, breaks=100))]

  # plot the phylogenetic tree
  pdf(paste(savePath, pic_name, ".pdf", sep=""))

  # set up the layout
  layout(matrix(c(1, 2), ncol = 2), widths = c(2.0, 1), heights = c(6, 1)) # 4:1 ratio for tree plot and color bar
  # plot the phylogenetic tree
  par(mar = c(2, 2, 2, 0.1) + 0.1) # Adjust margins to fit the plot nicely
  plot(phy_tree, show.tip.label = FALSE, edge.color = "grey")
  nodelabels(pch = 21, bg = node_colors, cex = 1.0, col = "grey")
  # add color bar legend
  par(mar = c(5, 1, 1, 4))  # Adjust margins as needed
  add_color_bar(scico_palette, min_val = pic_range[1], max_val = pic_range[2], title = "PIC Values", num_ticks = 5, tick_cex = 0.8)
  dev.off()
  
  # print out the names of the all the nodes with the 5 highest PIC values as well as their daughter tips
  # get the node numbers of the highest PIC values (ignore NaNs)
  top <- head(order(pic_normalized, decreasing = TRUE, na.last = NA), 5)
  topDescendants <- Descendants(phy_tree_phangorn, top)
  # get all the descendants
  for (i in 1:5) {
    print(paste(pic_name, ":", phy_tree$node.label[top[i]]))
  }
}