suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(fda))
suppressMessages(library(VGAM))

# Data frame storing functional principal components
# Note that PCs output later depend on this
curve_pca <- readRDS('curve_pca.rds')

# Convenience function for outputting gumbel-based GAM
gev_glm <- function(maxs, sigmas, backgrounds){
  df <- data.frame(ms = maxs, ss = sigmas, bb = backgrounds)
  b <- vglm(ms ~ ss*bb, gumbel(lscale = "identitylink"), data=df)
  return(coef(b, matrix=T))
}

get_pc_scores <- function(events){
  # set up the bspline basis that will be used for calculating fPCA scores
  curvebasis <- create.bspline.basis(rangeval = c(0,6.5),
                                     nbasis = 10, norder = 4)
  curvePar <- fdPar(curvebasis, 2, 1)
  
  smoother <- function(x){
    return(ksmooth(1:length(x), x, "normal", 3, n.points=length(x))$y)
  }
  
  
  ## Puffs
  # First, we calculate radial intensities for each puff
  events <- events %>%
    group_by(particle) %>%
    mutate(relframe = frame - min(frame)) %>%
    mutate(radius = sqrt(x^2 + y^2)) %>%
    arrange(particle, relframe, radius) %>%
    ungroup()
  
  # next, subtract out an estimate of initial background intensity
  # and scale from 0 to 1 so that all puffs have the same peak intensity
  # (to prevent big differences appearing between bright vs. dim puffs)
  events <- events %>%
    group_by(particle, relframe) %>%
    mutate(fitted_intens = predict(smooth.spline(radius, intensity), radius)$y) %>%
    ungroup() %>%
    group_by(particle) %>%
    mutate(intensity = intensity - mean(fitted_intens[relframe < 3]),
           fitted_intens = fitted_intens - mean(fitted_intens[relframe < 3]),
           intensity = (max(fitted_intens) - intensity)/(max(fitted_intens) - min(fitted_intens)),
           fitted_intens = (max(fitted_intens) - fitted_intens)/(max(fitted_intens) - min(fitted_intens))) %>%
          # intensity = 40*intensity/max(fitted_intens),
          # fitted_intens = 40*fitted_intens/max(fitted_intens)) %>%
    ungroup()
  
  # now we extract the radius and intensity information into two matrices 
  # (events_dist and events_intens, respectively); this is just re-formatting to get
  # the data into the right form for calculating pca scores
  events_intens <- events %>% 
    select(particle, relframe, radius, intensity) %>%
    group_by(particle, relframe) %>%
    arrange(particle, relframe, radius) %>%
    ungroup() %>%
    mutate(relPoint = rep(1:81, nrow(events)/81)) %>%
    select(-radius) %>%
    spread(relPoint, intensity) %>%
    select(-c(particle, relframe)) %>%
    as.matrix()
  
  events_dist <- events %>% 
    select(particle, relframe, radius) %>%
    group_by(particle, relframe) %>%
    arrange(particle, relframe, radius) %>%
    ungroup() %>%
    mutate(relPoint = rep(1:81, nrow(events)/81)) %>%
    spread(relPoint, radius) %>% 
    select(-c(particle, relframe)) %>%
    as.matrix()
  
  # calculate basis coefficients for the observed radial intensity profiles
  curvefd_events <- smooth.basis(t(events_dist), t(events_intens), curvePar)$fd
  
  # using the basis coefficients, calculate fpca scores for each event
  # based on the imported functional principal components
  # (we're not estimating principal components here, just projecting the observed data 
  # onto pre-calculated principal components)
  curvefd_events$coefs <- sweep(curvefd_events$coefs, 1, curve_pca$meanfd$coefs)
  events_pca_scores <- inprod(curvefd_events, curve_pca$harmonics)
  
  
  # match the scores to event info to get score data into nice format
  scoredf_events <- events %>%
    select(frame, particle) %>%
    distinct() %>%
    cbind(events_pca_scores)
  
  names(scoredf_events) <- c("frame", "particle", "s1", "s2", "s3")
  
  scoredf_events <- scoredf_events %>%
    group_by(particle) %>%
    mutate(smooth1 = smoother(s1),
           smooth2 = smoother(s2),
           smooth3 = smoother(s3))
  
  return(scoredf_events)
}