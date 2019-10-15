suppressMessages(library(geometry))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(fda))
suppressMessages(library(MASS))
select <- dplyr::select

### Functions for Scoring ###
#############################

### Main Scoring Functions ###

get_pc_scores <- function(events, fpca_obj = puff_pca) {
  pca_scores <- as.data.frame(calc_scores_1d(events, fpca_obj))
  colnames(pca_scores) <- c("s1", "s2", "s3")

  puff_scores <- events %>%
    select(xy, intensity, frame, particle, cell) %>%
    spread(xy, intensity) %>%
    select(cell, particle, frame) %>%
    distinct() %>%
    cbind(pca_scores) %>%
    mutate(smooth1 = ksmooth(frame, s1, bandwidth = 10, x.points=frame)$y,
           smooth2 = ksmooth(frame, s2, bandwidth = 10, x.points=frame)$y)
  return(puff_scores)
}

get_cca_scores <- function(puff_scores, fcca_obj = ccafd,
                           mean1 = puffmean1, mean2 = puffmean2) {
  curvefd_puff <- curvefd_2d_01(puff_scores)
  ccafdPar <- fdPar(curvefd_puff, 2, 1)
  cca_scores <- calc_cca_scores(curvefd_puff, fcca_obj, mean1, mean2)
  return(cca_scores)
}

get_features <- function(intensities, frame_length = 0.02) {
  events <- prepare_data(intensities)

  puff_scores <- get_pc_scores(events)

  cca_scores <- get_cca_scores(puff_scores)

  fluor_scores <- intensities %>%
    group_by(particle) %>%
    summarize(deltaf = max(intensity)/min(intensity))

  tau_scores <- intensities %>%
    filter(x == 0, y == 0) %>%
    group_by(particle) %>%
    summarize(tau = calc_tau(intensity) * frame_length)

  puff_features <- puff_scores %>%
    group_by(cell, particle) %>%
    summarize(conv_perim = convhulln(cbind(s1, s2), options="FA")$area,
              conv_area = convhulln(cbind(s1, s2), options="FA")$vol,
              lifetime_s = n()*frame_length,
              randomness_s1 = sum(abs(s1 - smooth1))/n(),
              randomness_s2 = sum(abs(s2 - smooth2))/n()) %>%
    bind_cols(cca_scores) %>%
    left_join(fluor_scores, by = 'particle') %>%
    left_join(tau_scores, by = 'particle')
  return(puff_features)
}


### Helper Functions ###

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
    # removing this so particles can be tracked back to initial events
    #mutate(particle = cumsum(c(1, diff(particle) != 0))) %>%
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

# calculate decay rate
# returns decay rate in number of frames
# if event never decays to more than half of its max difference, returns 0
calc_tau <- function(intensity_trace){
  peak_idx <- which.max(intensity_trace)
  after_peak <- tail(intensity_trace, -peak_idx)

  max_f <- max(intensity_trace)
  min_f <- min(intensity_trace)
  diff_f <- max_f - min_f
  tau <- 0

  for (val in after_peak){
    if ((max_f-val) >= 0.5*diff_f) {
      tau <- tau + 1
      break
    } else {
      tau <- tau+1
    }
  }

  if (tau >= length(after_peak)) {
    return(0)
  } else {
    return(tau)
  }
}
