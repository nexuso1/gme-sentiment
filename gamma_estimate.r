library("poweRlaw")
library("igraph")
library("stringr")
library("dplyr")
library(tidyverse)
# For twitter use "Networks/twitter_network_names.csv"
enron <- read.csv("Networks/enron_network.csv")
g <- graph_from_data_frame(enron[, c("source", "target")])
results = list() # List holding all bootstrap results
degree_types = c("in", "out", "all")
n_sims = 1000 # Number of bootstrap simulations
for (i in 1:length(degree_types)){
  # Li
  results[[i]] = list()
  m = degree_types[i]
  # Gather degrees
  degrees = degree(g, mode = m)
  # Throw out 0 values
  cleaned = {function(x) x[x!=0]} (degrees)
  # Fit the power law
  pl = displ$new(cleaned)
  # Estimate xmin
  xmin = estimate_xmin(pl)
  # Update the pl object
  pl$setXmin(xmin)
  
  # Bootstrap the dataset to get deviations and p values
  bs = bootstrap(pl, no_of_sims = n_sims, threads = 12, seed = 42)
  
  # Xmin standard deviation
  xmin_std = sd(bs$bootstraps[, 2])
  
  # Scaling parameter standard deviation
  sp_std = sd(bs$bootstraps[, 3])
  
  # Hypothesis testing with bootstrapping
  bs_p = bootstrap_p(pl, no_of_sims = n_sims, seed = 42, threads = 12, xmax = 500)
  p = bs_p$p
  
  # Save results
  results[[i]][[1]] <- bs
  results[[i]][[2]] <- bs_p
  results[[i]][[3]] <- pl
  
  plot(pl, main = paste("Distribution of", m, "degrees of the Enron network"),
       xlab = "Degree", ylab = "p(x)")
  lines(pl, col = 2)
  
  hist(bs$bootstraps[, 2], breaks = 20, xlab = "Degree", ylab = "Frequency", main = str_to_title(paste(m, "degrees, xmin bootstrap histogram")))
  hist(bs$bootstraps[, 3], breaks = 20, xlab = "gamma", ylab = "Frequency", main = str_to_title(paste(m, "degrees, gamma bootstrap histogram")))
  cat(m, "Estimated xmin:", pl$xmin, "std:", xmin_std, "Estimated gamma:", pl$pars[1], "std:", sp_std, "p value:", p)
}

# Create a singular plot for all degree distributions
par(mfcol=c(1,3))
for (i in 1:length(degree_types)) { 
  plot(results[[i]][[3]], main= str_to_title(paste(degree_types[i], "degrees", sep="-")),
       xlab = "Degree", ylab = "p(x)", cex.axis = 2, cex.lab = 2, cex.main = 2, cex = 2)
  lines(results[[i]][[3]], col=2, lwd = 3)
}

for(i in 1:length(degree_types)) { 
  hist(results[[i]][[1]]$bootstraps[, 2], breaks = 20, xlab = "Degree", ylab = "Frequency", 
       main = paste(str_to_title(paste(degree_types[i], "degrees", sep="-")), "k_min", sep = ", "),
       cex.axis=2, cex.lab = 2, cex.main = 2)
}

for(i in 1:length(degree_types)) { 
  hist(results[[i]][[1]]$bootstraps[, 3], breaks = 20, xlab = "gamma", ylab = "Frequency", 
       main = paste(str_to_title(paste(degree_types[i], "degrees", sep="-")), "gamma", sep = ", "),
       cex.axis=2, cex.lab = 2, cex.main = 2)
}