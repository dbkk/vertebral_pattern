# R script to double check the Pearson correlation after doing PIC for the main intergenic results

# load libraries
library(ape)
library(phytools)

# set path
basePath <- "./"
savePath <- paste(basePath, "plots/picWithApe/", sep="")
# make this directory if it doesn't already exist
dir.create(savePath, showWarnings = FALSE)

# make the list of data/tree files for testing and plotting
allList <- list(
  c("intergenicData_B9_C_full"),
  c("intergenicData_B9_T_full"),
  c("intergenicData_B9_L_full"),
  c("intergenicData_B9_S_full"),
  c("intergenicData_B9_Ca_mammals"),
  c("intergenicData_B4_T_reptiles"),
  c("intergenicData_A11_Ca_amphibians"),
  c("intergenicData_D1_T_amphibians")
)

# loop through and do the PIC on each
for (name in allList) {
 
  # load the tree
  tree <- read.tree(paste(basePath, "intergenic/", name, "_tree.nwk", sep=""))
  # load the data
  data <- read.csv(paste(basePath, "intergenic/", name, ".csv", sep=""))
  # get the node name as the last "_" in the name
  nodeName <- unlist(strsplit(as.character(name), "_"))[length(unlist(strsplit(as.character(name), "_")))]
  # extract the trait names
  trait_names <- names(data)[2:3]
  trait1_name <- trait_names[1]
  trait2_name <- trait_names[2]
  # filter out those with either NaNs in trait1 or trait2
  data <- data[!is.na(data[[trait1_name]]) & !is.na(data[[trait2_name]]), ]
  # reduce the tree to only those species in the data
  tree <- drop.tip(tree, tree$tip.label[!tree$tip.label %in% data$treeSpecies])
  # make sure the species names in your tree and data match
  species_order <- match(tree$tip.label, data$treeSpecies)
  trait1_data <- data[[trait1_name]][species_order]
  trait2_data <- data[[trait2_name]][species_order]
  # make sure that the species order is correct
  if(!is.null(species_order)) {
    valid_data <- !is.na(species_order)
    trait1_data <- trait1_data[valid_data]
    trait2_data <- trait2_data[valid_data]
    tree <- drop.tip(tree, tree$tip.label[!valid_data])
  }
  # adjust zero-length branches
  tree$edge.length[tree$edge.length == 0] <- 1
  # calculate the phylogenetic independent contrasts for each trait and filter out NA values
  pic_trait1 <- pic(trait1_data, tree, scaled = TRUE)
  pic_trait2 <- pic(trait2_data, tree, scaled = TRUE)
  valid_pics <- complete.cases(pic_trait1, pic_trait2)
  pic_trait1 <- pic_trait1[valid_pics]
  pic_trait2 <- pic_trait2[valid_pics]
  # analyze the contrasts (e.g., correlation and p-value)
  cor_test_result <- cor.test(pic_trait1, pic_trait2)
  # print the results
  print(paste("Correlation between", trait1_name, "and", trait2_name, ":", cor_test_result$estimate))
  print(paste("p-value:", cor_test_result$p.value))
  # plot
  pdf(paste(savePath, trait1_name, "_", trait2_name, "_", nodeName, "_pic_plot.pdf", sep=""))
  plot(pic_trait1, pic_trait2, main=paste("PIC of", trait1_name, "vs", trait2_name, "\n r = ", as.character(round(cor_test_result$estimate, digits = 3)), ", p = ", sprintf("%.2e", cor_test_result$p.value)), xlab=paste("PIC of", trait1_name), ylab=paste("PIC of", trait2_name))
  abline(lm(pic_trait2 ~ pic_trait1), col="red")  # add a regression line
  dev.off()
}
