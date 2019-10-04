import rpy2
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
numpy2ri.activate()
pandas2ri.activate()

import importlib.resources as pkg_resources

# This loads in the stored principal components and CCA objects needed to conduct analysis
# puff_pca: the PC space to project puffs into
# ccafd: the CCA object used to calculate new CCA scores
# puffmeans: mean values of puffs used to fit the above two objects
with pkg_resources.path('lib', 'puff_pca.rds') as filepath:
    rpy2.robjects.r("puff_pca <- readRDS('" + str(filepath) + "')")

with pkg_resources.path('lib', 'ccafd.rds') as filepath:
    rpy2.robjects.r("ccafd <- readRDS('" + str(filepath) + "')")

with pkg_resources.path('lib', 'puffmeans.Rdata') as filepath:
    rpy2.robjects.r["load"](str(filepath))

with pkg_resources.path('lib', 'analysis.R') as filepath:
    rpy2.robjects.r["source"](str(filepath))

get_pc_scores = rpy2.robjects.globalenv['get_pc_scores']
get_features = rpy2.robjects.globalenv['get_features']
