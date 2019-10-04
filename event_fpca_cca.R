library(dplyr)
library(tidyr)
library(ggplot2)
library(fda)
library(MASS)
select <- dplyr::select

########################
### Helper Functions ###
########################

# scale intensity across events so that event differences are not
# dominated by differences in brightness
adjust_intensity <- function(intensity, fitted_intens){
  if(max(fitted_intens) == min(fitted_intens)){
    intensity = rep(0, length(intensity))
  } else {
    intensity = (intensity - min(fitted_intens))/(max(fitted_intens) - min(fitted_intens))
  }
  return(intensity)
}

# prepare data from particle tracking output for fpca and further analysis 
# assumes event_intensities has columns particle, frame, x, y 
prepare_data <- function(event_intensities){
  if(!("cell" %in% colnames(event_intensities))){
    event_intensities <- event_intensities %>%
      mutate(cell = 1)
  }
  event_intensities <- event_intensities %>%
    distinct() %>%
    arrange(particle, frame) %>%
    mutate(particle = cumsum(c(1, diff(particle) != 0))) %>%
    mutate(radius = sqrt(x^2 + y^2)) %>%
    group_by(particle, frame) %>%
    mutate(fitted_intens = predict(smooth.spline(radius, intensity), radius)$y) %>%
    ungroup() %>%
    group_by(particle) %>%
    mutate(intensity = adjust_intensity(intensity, fitted_intens)) %>%
    ungroup() %>%
    unite("xy", c("x", "y"), sep=',')
  
  return(event_intensities)
}

# fit functional pca on a collection of events
# events should be in same format as output from prepare_data
# returns a functional data object (see fda package) that 
# represents the fpca fit
fit_fpca_1d <- function(events){
  puff_intens_mat <- events %>%
    select(xy, intensity, frame, particle, cell) %>%
    spread(xy, intensity) %>%
    select(-c(frame, particle, cell)) %>%
    as.data.frame() %>%
    as.matrix()
  
  puff_dist <- events %>%
    select(xy, frame, particle, cell, radius) %>%
    spread(xy, radius) %>%
    select(-c(particle, frame, cell)) %>%
    as.matrix()
  
  curvebasis <- create.bspline.basis(rangeval = c(0,6),
                                     nbasis = 10, norder = 4)
  curvePar <- fdPar(curvebasis, 2, 1)
  
  curvefd <- smooth.basis(t(puff_dist), t(puff_intens_mat), curvePar)$fd
  
  puff_pca <- pca.fd(curvefd, nharm=3, curvePar)
  return(puff_pca)
}

# calculate scores for a collection of events, from an 
# existing fpca fit
# allows fpca fit and fpca score calculation to be done on 
# different sets of events
# 'events' should look like output of prepare_data
# fpca_obj should look like output of fit_fpca_1d
calc_scores_1d <- function(events, fpca_obj, nharm=3, rangeval=c(0,6),
                           nbasis=10, norder=4, deriv=2, pen=1){
  
  puff_intens_mat <- events %>%
    select(xy, intensity, frame, particle, cell) %>%
    spread(xy, intensity) %>%
    select(-c(frame, particle, cell)) %>%
    as.data.frame() %>%
    as.matrix()
  
  puff_dist <- events %>%
    select(xy, frame, particle, cell, radius) %>%
    spread(xy, radius) %>%
    select(-c(particle, frame, cell)) %>%
    as.matrix()
  
  curvebasis <- create.bspline.basis(rangeval = c(0,6),
                                     nbasis = 10, norder = 4)
  curvePar <- fdPar(curvebasis, 2, 1)
  
  curvefd <- smooth.basis(t(puff_dist), t(puff_intens_mat), curvePar)$fd
  
  nrep <- dim(curvefd$coefs)[2]
  harmfd <- fpca_obj$harmonics
  
  harmscr <- array(0, c(nrep, nharm))
  coefarray <- sweep(curvefd$coefs, 1, fpca_obj$meanfd$coefs)
  harmcoefarray <- harmfd$coefs
  fdobjj <- fd(as.matrix(coefarray), curvefd$basis)
  harmfdj <- fd(as.matrix(harmcoefarray), curvefd$basis)
  harmscr <- inprod(fdobjj, harmfdj)
  
  return(harmscr)
}

# creates a functional data object for a collection of 
# 2D fpca score paths
curvefd_2d_01 <- function(scores, remove_start = 0){
  scores <- scores %>%
    group_by(particle) %>%
    mutate(s1c = s1 - mean(s1[1:4]),
           s2c = s2 - mean(s2[1:4]),
           relframe = frame - min(frame),
           relframe01 = (relframe - remove_start)/max(relframe - remove_start)) %>%
    ungroup() %>%
    mutate(s1c = scale(s1c),
           s2c = scale(s2c)) %>%
    filter(relframe01 >= 0)
  
  num_events <- scores %>%
    pull(particle) %>%
    unique() %>%
    length()
  
  num_basis <- 10
  basis_order <- 4
  basis_range <- c(0, 1)
  
  curvebasis <- create.bspline.basis(rangeval = basis_range,
                                     nbasis = num_basis, norder = basis_order)
  curvePar <- fdPar(curvebasis, 0, 1e-8)
  
  coef_mat <- array(0, c(num_basis, num_events, 2))
  particle_ids <- scores %>%
    pull(particle) %>%
    unique()
  
  for(i in 1:num_events){
    id <- particle_ids[i]
    particle_t <- scores %>%
      filter(particle == id) %>%
      pull(relframe01)
    particle_s <- scores %>%
      filter(particle == id) %>%
      select(s1c, s2c) %>%
      as.matrix() %>%
      array(c(length(particle_t), 1, 2))
    curvefd <- smooth.basis(particle_t, particle_s, curvePar)$fd
    coef_mat[,i,] <- curvefd$coefs
  }
  
  curvefd$coefs <- coef_mat
  return(curvefd)
}

# calculate cca scores
# allows cca to be fit on one set of events, then scores 
# calculated on a new set of events
calc_cca_scores <- function(fd_obj, cca_obj, mean1, mean2){
  canwtcoef1 <- ccafd$ccawtfd1$coefs
  canwtcoef2 <- ccafd$ccawtfd2$coefs
  
  fdobj1 <- fd_obj[,1]
  fdobj2 <- fd_obj[,2]
  
  fdobj1$coefs <- sweep(fdobj1$coefs, 1, mean1)
  fdobj2$coefs <- sweep(fdobj2$coefs, 1, mean2)
  
  coef1 <- fdobj1$coefs
  coef2 <- fdobj2$coefs
  basisobj1 <- fdobj1$basis
  basisobj2 <- fdobj2$basis
  Jmat1 <- eval.penalty(basisobj1, 0)
  Jmat2 <- eval.penalty(basisobj2, 0)
  Jx <- t(Jmat1 %*% coef1)
  Jy <- t(Jmat2 %*% coef2)
  
  out1 <- Jx %*% canwtcoef1
  out2 <- Jy %*% canwtcoef2
  output <- data.frame(cbind(out1, out2))
  colnames(output) <- c("cc11", "cc12", "cc13", "cc21", "cc22", "cc23")
  return(output)
}



####################
### Read in Data ###
####################

puff_clusters <- read.csv('data/0IYVHRNA_clusters_Zara.csv') %>%
  rbind(read.csv('data/19CC75ZU_clusters_Zara.csv')) %>%
  rbind(read.csv('data/2TZWB6CN_clusters_Zara.csv')) %>%
  rbind(read.csv('data/3W70AV4V_clusters_Zara.csv'))

puff_events <- read.csv('data/0IYVHRNA.tif_puff_intensities.csv') %>%
  mutate(cell = 1) %>%
  rbind(read.csv('data/19CC75ZU.tif_puff_intensities.csv') %>% mutate(cell = 2)) %>%
  rbind(read.csv('data/2TZWB6CN.tif_puff_intensities.csv') %>% mutate(cell = 3)) %>%
  rbind(read.csv('data/3W70AV4V.tif_puff_intensities.csv') %>% mutate(cell = 4)) %>%
  prepare_data()

nonpuff_events <- read.csv('data/0IYVHRNA.tif_nonpuff_intensities.csv') %>%
  mutate(cell = 1) %>%
  rbind(read.csv('data/19CC75ZU.tif_nonpuff_intensities.csv') %>% mutate(cell = 2)) %>%
  rbind(read.csv('data/2TZWB6CN.tif_nonpuff_intensities.csv') %>% mutate(cell = 3)) %>%
  rbind(read.csv('data/3W70AV4V.tif_nonpuff_intensities.csv') %>% mutate(cell = 4)) %>%
  prepare_data()




#############################
### Get Radial PCA Scores ###
#############################

puff_pca <- fit_fpca_1d(puff_events)
pca_scores <- as.data.frame(calc_scores_1d(puff_events, puff_pca))
colnames(pca_scores) <- c("s1", "s2", "s3")

puff_scores <- puff_events %>%
  select(xy, intensity, frame, particle, cell) %>%
  spread(xy, intensity) %>%
  select(cell, particle, frame) %>%
  distinct() %>%
  cbind(pca_scores)


nonpuff_scores <- as.data.frame(calc_scores_1d(nonpuff_events, puff_pca))
colnames(nonpuff_scores) <- c("s1", "s2", "s3")
nonpuff_scores <- nonpuff_events %>%
  select(-c(X, radius, fitted_intens)) %>%
  spread(xy, intensity) %>%
  select(cell, particle, frame) %>%
  distinct() %>%
  cbind(nonpuff_scores)


###########################
### Basis Fit 2D Curves ###
###########################

curvefd_puff <- curvefd_2d_01(puff_scores)
curvefd_nonpuff <- curvefd_2d_01(nonpuff_scores)


############################
### Calculate CCA Scores ###
############################

ccafdPar <- fdPar(curvefd_puff, 2, 1)
ccafd <- cca.fd(curvefd_puff[,1], curvefd_puff[,2], ncan=3, ccafdPar, ccafdPar)

puffmean1 <- apply(as.array(curvefd_puff[,1]$coefs), 1, mean) 
puffmean2 <- apply(as.array(curvefd_puff[,2]$coefs), 1, mean)

cca_puff_scores <- calc_cca_scores(curvefd_puff, ccafd, puffmean1, puffmean2)
cca_nonpuff_scores <- calc_cca_scores(curvefd_nonpuff, ccafd, puffmean1, puffmean2)



########################
### Plot CCA Results ###
########################

# we can see there is some separation between puffs and nonpuffs
# cca scores could be useful in a classifier; 
# fit cca on a training set of known puffs
# then calculate cca scores on a new set of unlabeled events

ggplot(data=cca_puff_scores, aes(x = cc11, y = cc21)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_point(data = cca_nonpuff_scores, color = "red", alpha = 0.5)



## Convex hull perimeter and area

library(geometry)

# calculate convex hull perimeter and convex hull area
puff_features <- puff_scores %>%
  group_by(cell, particle) %>%
  summarize(conv_perim = convhulln(cbind(s1, s2), options="FA")$area,
          conv_area = convhulln(cbind(s1, s2), options="FA")$vol)
