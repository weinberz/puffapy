suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(VGAM))

#########################
# Functions for Detection
#########################

# Convenience function for outputting gumbel-based GAM
gev_glm <- function(maxs, sigmas, backgrounds){
  df <- data.frame(ms = maxs, ss = sigmas, bb = backgrounds)
  b <- vglm(ms ~ ss*bb, gumbel(lscale = "identitylink"), data=df)
  return(coef(b, matrix=T))
}
